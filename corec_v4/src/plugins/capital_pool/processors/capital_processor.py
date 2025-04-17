#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/capital_pool/processors/capital_processor.py
"""
capital_processor.py
Gestiona el capital compartido de usuarios, asigna fondos para trading, y ajusta fases dinámicamente.
"""

from ....core.processors.base import ProcesadorBase
from ....core.entidad_base import Event
from ....utils.logging import logger
from ..utils.db import CapitalDB
import json
import zstandard as zstd
from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio

class CapitalProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("CapitalProcessor")
        self.min_contribution = config.get("capital_config", {}).get("min_contribution", 100)
        self.max_active_ratio = config.get("capital_config", {}).get("max_active_ratio", 0.6)
        self.phases = config.get("capital_config", {}).get("phases", [])
        self.circuit_breaker = config.get("config", {}).get("circuit_breaker", {})
        self.plugin_db = None
        self.pool = 0.0  # Capital total del pool
        self.active_capital = 0.0  # Capital activo en trades
        self.users = {}  # Contribuciones por usuario
        self.macro_context = {}  # Contexto macro
        self.failure_count = 0
        self.breaker_tripped = False
        self.breaker_reset_time = None

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        # Inicializar base de datos
        self.plugin_db = CapitalDB(self.db_config)
        if not await self.plugin_db.connect():
            self.logger.warning("No se pudo conectar a capital_db, usando almacenamiento temporal")

        # Cargar estado inicial
        if self.plugin_db and self.plugin_db.conn:
            self.pool = await self.plugin_db.get_pool_total()
            self.users = await self.plugin_db.get_users()
            self.active_capital = await self.plugin_db.get_active_capital()

        # Iniciar gestión de capital
        asyncio.create_task(self.manage_pool())
        self.logger.info("CapitalProcessor inicializado")

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
            await self.nucleus.publicar_alerta({"tipo": "circuit_breaker_tripped", "plugin": "capital_pool"})

    async def get_current_phase(self) -> Dict[str, Any]:
        for phase in self.phases:
            if phase["min"] <= self.pool < phase["max"]:
                return phase
        return self.phases[-1]  # Última fase por defecto

    async def add_contribution(self, user_id: str, amount: float) -> bool:
        if amount < self.min_contribution:
            self.logger.warning(f"Contribución de {user_id} ({amount}) menor al mínimo ({self.min_contribution})")
            return False
        try:
            if user_id not in self.users:
                self.users[user_id] = 0.0
            self.users[user_id] += amount
            self.pool += amount
            if self.plugin_db and self.plugin_db.conn:
                await self.plugin_db.save_contribution(user_id, amount, datetime.utcnow().timestamp())
                await self.plugin_db.update_pool(self.pool)
            self.logger.info(f"Contribución de {user_id}: {amount}, pool total: {self.pool}")
            return True
        except Exception as e:
            self.logger.error(f"Error añadiendo contribución: {e}")
            await self.register_failure()
            return False

    async def process_withdrawal(self, user_id: str, amount: float) -> bool:
        if user_id not in self.users or self.users[user_id] < amount:
            self.logger.warning(f"Retiro de {user_id} ({amount}) excede contribución disponible")
            return False
        try:
            self.users[user_id] -= amount
            self.pool -= amount
            if self.users[user_id] <= 0:
                del self.users[user_id]
            if self.plugin_db and self.plugin_db.conn:
                await self.plugin_db.save_withdrawal(user_id, amount, datetime.utcnow().timestamp())
                await self.plugin_db.update_pool(self.pool)
            self.logger.info(f"Retiro de {user_id}: {amount}, pool total: {self.pool}")
            return True
        except Exception as e:
            self.logger.error(f"Error procesando retiro: {e}")
            await self.register_failure()
            return False

    async def update_pool_from_results(self, result: Dict[str, Any]) -> None:
        profit = result.get("profit", 0)
        if profit != 0:
            self.pool += profit
            self.active_capital -= result.get("quantity", 0) * result.get("price", 0)
            if self.plugin_db and self.plugin_db.conn:
                await self.plugin_db.update_pool(self.pool)
                await self.plugin_db.update_active_capital(self.active_capital)
            self.logger.info(f"Pool actualizado con profit: {profit}, total: {self.pool}")

    async def assign_capital(self) -> None:
        if not await self.check_circuit_breaker():
            return

        phase = await self.get_current_phase()
        risk_per_trade = phase["risk_per_trade"]
        max_active = self.pool * self.max_active_ratio

        # Ajustar riesgo con macro datos
        risk_adjustment = 1.0
        if self.macro_context.get("vix_price", 0) > 20:
            risk_adjustment = 0.5  # Reducir riesgo en alta volatilidad

        available_capital = max(0, max_active - self.active_capital)
        trade_amount = min(available_capital, self.pool * risk_per_trade * risk_adjustment)

        if trade_amount > 0:
            capital_data = {
                "trade_amount": trade_amount,
                "phase": phase["name"],
                "risk_per_trade": risk_per_trade,
                "timestamp": datetime.utcnow().timestamp()
            }
            datos_comprimidos = zstd.compress(json.dumps(capital_data).encode())
            await self.redis_client.xadd("capital_data", {"data": datos_comprimidos})
            self.active_capital += trade_amount
            if self.plugin_db and self.plugin_db.conn:
                await self.plugin_db.update_active_capital(self.active_capital)
            self.logger.info(f"Asignado {trade_amount} para trading, fase: {phase['name']}")

    async def manage_pool(self):
        while True:
            await self.assign_capital()
            await asyncio.sleep(300)  # Cada 5 minutos

    async def manejar_evento(self, event: Event) -> None:
        try:
            if event.canal == "trading_results":
                result = json.loads(zstd.decompress(event.datos["data"]))
                await self.update_pool_from_results(result)
            elif event.canal == "macro_data":
                self.macro_context = event.datos
                self.logger.info("Datos macro recibidos: %s", self.macro_context)
        except Exception as e:
            self.logger.error(f"Error procesando evento: {e}")
            await self.register_failure()

    async def detener(self):
        if self.plugin_db:
            await self.plugin_db.disconnect()
        self.logger.info("CapitalProcessor detenido")