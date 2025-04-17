#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/alert_manager/processors/alert_processor.py
"""
alert_processor.py
Gestiona y clasifica alertas con contexto de DXY y system_analyzer, enviando notificaciones a email y Discord.
"""

from ....core.processors.base import ProcesadorBase
from ....core.entidad_base import Event
from ....utils.logging import logger
from ..utils.db import AlertDB
import zstandard as zstd
import json
from typing import Dict, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
import psycopg2

class AlertProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("AlertProcessor")
        self.plugin_db = None
        self.circuit_breaker = config.get("config", {}).get("circuit_breaker", {})
        self.failure_count = 0
        self.breaker_tripped = False
        self.breaker_reset_time = None
        self.thresholds = config.get("alert_config", {}).get("thresholds", {"vix": 20})
        self.email_config = config.get("notification_config", {}).get("email", {})
        self.discord_webhook = config.get("notification_config", {}).get("discord", {}).get("webhook_url", "")
        self.data_cache = {}

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.plugin_db = AlertDB(self.db_config)
        if not await self.plugin_db.connect():
            self.logger.warning("No se pudo conectar a alert_db")
            await self.nucleus.publicar_alerta({"tipo": "db_connection_error", "plugin": "alert_manager", "message": "No se pudo conectar a alert_db"})
        self.logger.info("AlertProcessor inicializado")

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
            await self.nucleus.publicar_alerta({"tipo": "circuit_breaker_tripped", "plugin": "alert_manager"})

    async def send_email(self, message: str, severity: str) -> bool:
        if not self.email_config or severity not in ["high", "medium"]:
            return False
        try:
            # Obtener usuarios con preferencia de email
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT email FROM users WHERE notification_preferences->>'email' = 'true'")
            emails = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()

            if not emails:
                return False

            # Configurar correo
            msg = MIMEText(f"[CoreC Alert - {severity.upper()}] {message}")
            msg['Subject'] = f"CoreC Alert: {severity.upper()}"
            msg['From'] = self.email_config.get("from_email")
            msg['To'] = ", ".join(emails)

            # Conectar al servidor SMTP
            with smtplib.SMTP(self.email_config.get("smtp_server"), self.email_config.get("smtp_port")) as server:
                server.starttls()
                server.login(self.email_config.get("username"), self.email_config.get("password"))
                server.send_message(msg)

            self.logger.info(f"Email enviado a {len(emails)} usuarios")
            return True
        except Exception as e:
            self.logger.error(f"Error enviando email: {e}")
            return False

    async def send_discord(self, message: str, severity: str) -> bool:
        if not self.discord_webhook or severity not in ["high", "medium"]:
            return False
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "content": f"**CoreC Alert - {severity.upper()}**\n{message}",
                    "username": "CoreC Bot"
                }
                async with session.post(self.discord_webhook, json=payload) as resp:
                    if resp.status == 204:
                        self.logger.info("Notificación enviada a Discord")
                        return True
                    else:
                        self.logger.error(f"Error enviando a Discord: {resp.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Error enviando a Discord: {e}")
            return False

    async def procesar_alerta(self, alerta: Dict[str, Any]) -> Dict[str, Any]:
        try:
            severity = "low"
            if alerta["tipo"] in ["circuit_breaker_tripped", "action_failed", "api_exception"]:
                severity = "high"
            elif alerta["tipo"] in ["dxy_change", "no_historical_data"]:
                severity = "medium" if abs(alerta.get("message", "").find("DXY cambió") > -1 and float(alerta["message"].split()[2]) > 1) else "low"
            elif alerta["tipo"] == "action_executed":
                severity = "medium"

            # Añadir contexto
            context = {}
            if "macro_data" in self.data_cache:
                context["dxy"] = self.data_cache["macro_data"].get("dxy_price", 0)
                context["dxy_change"] = self.data_cache["macro_data"].get("dxy_change_percent", 0)
            if "system_insights" in self.data_cache:
                context["recommendations"] = self.data_cache["system_insights"].get("recommendations", [])

            alerta["severity"] = severity
            alerta["context"] = context
            alerta["timestamp"] = alerta.get("timestamp", datetime.utcnow().timestamp())

            datos_comprimidos = zstd.compress(json.dumps(alerta).encode())
            await self.redis_client.xadd("alert_data", {"data": datos_comprimidos})

            if self.plugin_db and self.plugin_db.conn:
                await self.plugin_db.save_alert(
                    alerta_id=f"alert_{alerta['timestamp']}",
                    tipo=alerta["tipo"],
                    message=alerta["message"],
                    severity=severity,
                    context=context,
                    timestamp=alerta["timestamp"]
                )

            # Enviar notificaciones externas
            if severity in ["high", "medium"]:
                await self.send_email(alerta["message"], severity)
                await self.send_discord(alerta["message"], severity)

            return {"estado": "ok", "alerta_id": f"alert_{alerta['timestamp']}"}
        except Exception as e:
            self.logger.error(f"Error procesando alerta: {e}")
            await self.register_failure()
            return {"estado": "error", "mensaje": str(e)}

    async def manejar_evento(self, event: Event) -> None:
        try:
            datos = json.loads(zstd.decompress(event.datos["data"]))
            if event.canal == "alertas":
                await self.procesar_alerta(datos)
            elif event.canal in ["macro_data", "system_insights"]:
                self.data_cache[event.canal] = datos
            elif event.canal == "alert_data" and datos.get("action") == "update_threshold":
                self.thresholds["vix"] = datos["vix_threshold"]
                self.logger.info(f"Umbral VIX actualizado a {self.thresholds['vix']}")
        except Exception as e:
            self.logger.error(f"Error manejando evento: {e}")
            await self.register_failure()
            await self.nucleus.publicar_alerta({"tipo": "event_error", "plugin": "alert_manager", "message": str(e)})

    async def detener(self):
        if self.plugin_db:
            await self.plugin_db.disconnect()
        self.logger.info("AlertProcessor detenido")