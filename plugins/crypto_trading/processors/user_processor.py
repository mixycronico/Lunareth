#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/user_processor.py
Gestiona usuarios, roles (user, admin, superadmin), autenticación con JWT, y preferencias de notificación.
"""
from corec.core import ComponenteBase, zstd, serializar_mensaje
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker
import jwt
import json
import asyncio
import bcrypt
from typing import Dict, Any
from datetime import datetime, timedelta

class UserProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("UserProcessor")
        self.plugin_db = TradingDB(self.config.get("db_config", {}))
        self.circuit_breaker = CircuitBreaker(
            self.config.get("user_config", {}).get("circuit_breaker", {}).get("max_failures", 3),
            self.config.get("user_config", {}).get("circuit_breaker", {}).get("reset_timeout", 900)
        )
        self.jwt_secret = self.config.get("user_config", {}).get("jwt_secret", "secure_secret")
        self.valid_roles = ["user", "admin", "superadmin"]
        self.permissions = {
            "user": ["view_reports", "contribute_pool", "receive_notifications"],
            "admin": ["view_reports", "contribute_pool", "receive_notifications", "manage_users", "configure_trading", "approve_operations"],
            "superadmin": ["view_reports", "contribute_pool", "receive_notifications", "manage_users", "configure_trading", "approve_operations", "manage_plugins", "configure_system", "view_audits"]
        }

    async def inicializar(self):
        await self.plugin_db.connect()
        self.logger.info("UserProcessor inicializado")

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

    async def hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    async def verify_password(self, password: str, hashed: str) -> bool:
        return bcrypt.checkpw(password.encode(), hashed.encode())

    async def generate_jwt(self, user_id: str, role: str) -> str:
        payload = {
            "user_id": user_id,
            "role": role,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")

    async def procesar_usuario(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        if not self.circuit_breaker.check():
            return {"estado": "error", "mensaje": "Circuit breaker activo"}
        try:
            action = datos.get("action")
            user_id = datos.get("user_id")
            requester_id = datos.get("requester_id")
            if action in ["register", "update", "delete", "update_preferences"]:
                if not await self.check_permission(requester_id, "manage_users"):
                    return {"estado": "error", "mensaje": f"Usuario {requester_id} no tiene permiso para {action}"}
            if action == "register":
                role = datos.get("role", "user")
                if role not in self.valid_roles:
                    return {"estado": "error", "mensaje": f"Rol inválido: {role}"}
                password = await self.hash_password(datos.get("password", ""))
                user_data = {
                    "user_id": user_id,
                    "email": datos.get("email"),
                    "password": password,
                    "name": datos.get("name"),
                    "role": role,
                    "notification_preferences": datos.get("notification_preferences", {"email": False}),
                    "created_at": datetime.utcnow().timestamp()
                }
                await self.plugin_db.save_user(**user_data)
                jwt_token = await self.generate_jwt(user_id, role)
                datos_comprimidos = zstd.compress(json.dumps(user_data).encode())
                mensaje = await serializar_mensaje(int(user_data["created_at"] % 1000000), self.canal, 0.0, True)
                await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
                return {"estado": "ok", "user_id": user_id, "jwt_token": jwt_token}
            elif action == "login":
                user = await self.plugin_db.get_user(user_id)
                if not user or not await self.verify_password(datos.get("password", ""), user["password"]):
                    return {"estado": "error", "mensaje": "Credenciales inválidas"}
                jwt_token = await self.generate_jwt(user_id, user["role"])
                return {"estado": "ok", "user_id": user_id, "jwt_token": jwt_token}
            elif action == "update":
                user_data = {
                    "user_id": user_id,
                    "email": datos.get("email"),
                    "password": await self.hash_password(datos["password"]) if datos.get("password") else None,
                    "name": datos.get("name"),
                    "role": datos.get("role"),
                    "notification_preferences": datos.get("notification_preferences")
                }
                if user_data["role"] and user_data["role"] not in self.valid_roles:
                    return {"estado": "error", "mensaje": f"Rol inválido: {user_data['role']}"}
                await self.plugin_db.update_user(**user_data)
                datos_comprimidos = zstd.compress(json.dumps(user_data).encode())
                mensaje = await serializar_mensaje(int(datetime.utcnow().timestamp() % 1000000), self.canal, 0.0, True)
                await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
                return {"estado": "ok", "user_id": user_id}
            elif action == "delete":
                await self.plugin_db.delete_user(user_id)
                await self.redis_client.delete(f"user:{user_id}")
                return {"estado": "ok", "user_id": user_id}
            elif action == "update_preferences":
                await self.plugin_db.update_user_preferences(
                    user_id=user_id,
                    notification_preferences=datos.get("notification_preferences")
                )
                datos_comprimidos = zstd.compress(json.dumps({"user_id": user_id, "action": "update_preferences"}).encode())
                mensaje = await serializar_mensaje(int(datetime.utcnow().timestamp() % 1000000), self.canal, 0.0, True)
                await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
                return {"estado": "ok", "user_id": user_id}
            return {"estado": "error", "mensaje": "Acción no soportada"}
        except Exception as e:
            self.logger.error(f"Error procesando usuario: {e}")
            self.circuit_breaker.register_failure()
            return {"estado": "error", "mensaje": str(e)}

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            if mensaje.get("tipo") == "user_data":
                result = await self.procesar_usuario(mensaje)
                if result["estado"] == "ok":
                    await self.redis_client.setex(f"user:{mensaje['user_id']}", 300, json.dumps(mensaje))
            elif mensaje.get("tipo") in ["trading_results", "capital_data"]:
                self.data_cache[mensaje["tipo"]] = mensaje
                self.logger.debug(f"Datos recibidos: {mensaje['tipo']}")
        except Exception as e:
            self.logger.error(f"Error manejando evento: {e}")
            self.circuit_breaker.register_failure()
            await self.nucleus.publicar_alerta({
                "tipo": "event_error",
                "plugin": "crypto_trading",
                "message": str(e)
            })

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("UserProcessor detenido")