#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/user_management/processors/user_processor.py
"""
user_processor.py
Gestiona usuarios, roles (user, admin, superadmin), autenticación con JWT, y preferencias de notificación.
"""

from ....core.processors.base import ProcesadorBase
from ....core.entidad_base import Event
from ....utils.logging import logger
from ..utils.db import UserDB
import zstandard as zstd
import json
from typing import Dict, Any
from datetime import datetime, timedelta
import jwt
import asyncio

class UserProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("UserProcessor")
        self.plugin_db = None
        self.circuit_breaker = config.get("config", {}).get("circuit_breaker", {})
        self.failure_count = 0
        self.breaker_tripped = False
        self.breaker_reset_time = None
        self.jwt_secret = config.get("jwt_config", {}).get("secret", "secure_secret")
        self.data_cache = {}
        self.valid_roles = ["user", "admin", "superadmin"]
        self.permissions = {
            "user": ["view_reports", "contribute_pool", "receive_notifications"],
            "admin": ["view_reports", "contribute_pool", "receive_notifications", "manage_users", "configure_trading", "approve_operations"],
            "superadmin": ["view_reports", "contribute_pool", "receive_notifications", "manage_users", "configure_trading", "approve_operations", "manage_plugins", "configure_system", "view_audits"]
        }

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.plugin_db = UserDB(self.db_config)
        if not await self.plugin_db.connect():
            self.logger.warning("No se pudo conectar a user_db")
            await self.nucleus.publicar_alerta({"tipo": "db_connection_error", "plugin": "user_management", "message": "No se pudo conectar a user_db"})
        self.logger.info("UserProcessor inicializado")

    async def check_circuit_breaker(self) -> bool:
        if self.breaker_tripped:
            now = datetime.utcnow()
            if now >= self.breaker_reset_time:
                self.breaker_tripped = False
                self.failure_count = 0
                self.breaker_reset_time = None
                self.logger.info("Circuit breaker reseteado")
            else:
                self.logger.warning("Circuit breaker activo hasta %s", self.breaker_reset_time)
                return False
        return True

    async def register_failure(self) -> None:
        self.failure_count += 1
        if self.failure_count >= self.circuit_breaker.get("max_failures", 3):
            self.breaker_tripped = True
            self.breaker_reset_time = datetime.utcnow() + timedelta(seconds=self.circuit_breaker.get("reset_timeout", 900))
            self.logger.error("Circuit breaker activado hasta %s", self.breaker_reset_time)
            await self.nucleus.publicar_alerta({"tipo": "circuit_breaker_tripped", "plugin": "user_management"})

    async def check_permission(self, user_id: str, permission: str) -> bool:
        try:
            user = await self.plugin_db.get_user(user_id)
            if not user:
                return False
            role = user.get("role", "user")
            return permission in self.permissions.get(role, [])
        except Exception as e:
            self.logger.error(f"Error verificando permiso: {e}")
            return False

    async def procesar_usuario(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        try:
            action = datos.get("action")
            user_id = datos.get("user_id")
            requester_id = datos.get("requester_id")  # ID del usuario que realiza la acción

            # Verificar permisos del solicitante
            if action in ["register", "update", "delete", "update_preferences"]:
                if not await self.check_permission(requester_id, "manage_users"):
                    return {"estado": "error", "mensaje": f"Usuario {requester_id} no tiene permiso para {action}"}

            if action == "register":
                role = datos.get("role", "user")
                if role not in self.valid_roles:
                    return {"estado": "error", "mensaje": f"Rol inválido: {role}"}
                user_data = {
                    "user_id": user_id,
                    "email": datos.get("email"),
                    "password": datos.get("password"),  # Debe ser hasheado en producción
                    "name": datos.get("name"),
                    "role": role,
                    "notification_preferences": datos.get("notification_preferences", {"email": False}),
                    "created_at": datetime.utcnow().timestamp()
                }
                if self.plugin_db and self.plugin_db.conn:
                    await self.plugin_db.save_user(**user_data)
                datos_comprimidos = zstd.compress(json.dumps(user_data).encode())
                await self.redis_client.xadd("user_data", {"data": datos_comprimidos})
                return {"estado": "ok", "user_id": user_id}

            elif action == "update":
                user_data = {
                    "user_id": user_id,
                    "email": datos.get("email"),
                    "password": datos.get("password"),
                    "name": datos.get("name"),
                    "role": datos.get("role"),
                    "notification_preferences": datos.get("notification_preferences")
                }
                if user_data["role"] and user_data["role"] not in self.valid_roles:
                    return {"estado": "error", "mensaje": f"Rol inválido: {user_data['role']}"}
                if self.plugin_db and self.plugin_db.conn:
                    await self.plugin_db.update_user(**user_data)
                datos_comprimidos = zstd.compress(json.dumps(user_data).encode())
                await self.redis_client.xadd("user_data", {"data": datos_comprimidos})
                return {"estado": "ok", "user_id": user_id}

            elif action == "delete":
                if self.plugin_db and self.plugin_db.conn:
                    await self.plugin_db.delete_user(user_id)
                await self.redis_client.delete(f"user:{user_id}")
                return {"estado": "ok", "user_id": user_id}

            elif action == "update_preferences":
                if self.plugin_db and self.plugin_db.conn:
                    await self.plugin_db.update_user_preferences(
                        user_id=user_id,
                        notification_preferences=datos.get("notification_preferences")
                    )
                return {"estado": "ok", "user_id": user_id}

            return {"estado": "error", "mensaje": "Acción no soportada"}
        except Exception as e:
            self.logger.error(f"Error procesando usuario: {e}")
            await self.register_failure()
            return {"estado": "error", "mensaje": str(e)}

    async def manejar_evento(self, event: Event) -> None:
        try:
            datos = json.loads(zstd.decompress(event.datos["data"]))
            if event.canal == "user_data":
                result = await self.procesar_usuario(datos)
                if result["estado"] == "ok":
                    await self.redis_client.setex(f"user:{datos['user_id']}", 300, json.dumps(datos))
            elif event.canal in ["trading_results", "capital_data"]:
                self.data_cache[event.canal] = datos
        except Exception as e:
            self.logger.error(f"Error manejando evento: {e}")
            await self.register_failure()
            await self.nucleus.publicar_alerta({"tipo": "event_error", "plugin": "user_management", "message": str(e)})

    async def detener(self):
        if self.plugin_db:
            await self.plugin_db.disconnect()
        self.logger.info("UserProcessor detenido")