#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/capital_processor.py
Gestiona el capital compartido de usuarios, asigna fondos para trading, y ajusta fases dinámicamente.
"""
from corec.core import ComponenteBase, zstd, serializar_mensaje
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker
import json
import asyncio
from typing import Dict, Any
from datetime import datetime, timedelta

class CapitalProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("CapitalProcessor")
        self.min_contribution = self.config.get("capital_config", {}).get("min_contribution", 100)
        self.max_active_ratio = self.config.get("capital_config", {}).get("max_active_ratio", 0.6)
        self.phases = self.config.get("capital_config", {}).get("phases", [])
        self.circuit_breaker = CircuitBreaker(
            self.config.get("capital_config", {}).get("circuit_breaker", {}).get("max_failures", 3),
            self.config.get("capital_config", {}).get("circuit_breaker", {}).get("reset_timeout", 900)
        )
        self.plugin_db = TradingDB(self.config.get("db_config", {}))
        self.pool = 0.0
        self.active_capital = 0.0
        self.users = {}
        self.macro_context = {}

    async def inicializar(self):
        await self.plugin_db.connect()
        self.pool = await self.plugin_db.get_pool_total()
        self.users = await self.plugin_db.get_users()
        self.active_capital = await self.plugin_db.get_active_capital()
        asyncio.create_task(self.manage_pool())
        self.logger.info("CapitalProcessor inicializado")

    async def get_current_phase(self) -> Dict[str, Any]:
        for phase in self.phases:
            if phase["min"] <= self.pool < phase["max"]:
                return phase
        return self.phases[-1]

    async def add_contribution(self, user_id: str, amount: float) -> bool:
        if not self.circuit_breaker.check():
            return False
        if amount < self.min_contribution:
            self.logger.warning(f"Contribución de {user_id} ({amount}) menor al mínimo ({self.min_contribution})")
            return False
        try:
            if user_id not in self.users:
                self.users[user_id] = 0.0
            self.users[user_id] += amount
            self.pool += amount
            await self.plugin_db.save_contribution(user_id, amount, datetime.utcnow().timestamp())
            await self.plugin_db.update_pool(self.pool)
            datos_comprimidos = zstd.compress(json.dumps({"user_id": user_id, "amount": amount, "action": "contribution"}).encode())
            mensaje = await serializar_mensaje(int(datetime.utcnow().timestamp() % 1000000), self.canal, amount, True)
            await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
            self.logger.info(f"Contribución de {user_id}: {amount}, pool total: {self.pool}")
            return True
        except Exception as e:
            self.logger.error(f"Error añadiendo contribución: {e}")
            self.circuit_breaker.register_failure()
            return False

    async def process_withdrawal(self, user_id: str, amount: float) -> bool:
        if not self.circuit_breaker.check():
            return False
        if user_id not in self.users or self.users[user_id] < amount:
            self.logger.warning(f"Retiro de {user_id} ({amount}) excede contribución disponible")
            return False
        try:
            self.users[user_id] -= amount
            self.pool -= amount
            if self.users[user_id] <= 0:
                del self.users[user_id]
            await self.plugin_db.save_withdrawal(user_id, amount, datetime.utcnow().timestamp())
            await self.plugin_db.update_pool(self.pool)
            datos_comprimidos = zstd.compress(json.dumps({"user_id": user_id, "amount": amount, "action": "withdrawal"}).encode())
            mensaje = await serializar_mensaje(int(datetime.utcnow().timestamp() % 1000000), self.canal, -amount, True)
            await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
            self.logger.info(f"Retiro de {user_id}: {amount}, pool total: {self.pool}")
            return True
        except Exception as e:
            self.logger.error(f"Error procesando retiro: {e}")
            self.circuit_breaker.register_failure()
            return False

    async def update_pool_from_results(self, result: Dict[str, Any]) -> None:
        if not self.circuit_breaker.check():
            return
        try:
            profit = result.get("profit", 0)
            if profit != 0:
                self.pool += profit
                self.active_capital -= result.get("quantity", 0) * result.get("price", 0)
                await self.plugin_db.update_pool(self.pool)
                await self.plugin_db.update_active_capital(self.active_capital)
                datos_comprimidos = zstd.compress(json.dumps({"action": "update_pool", "profit": profit}).encode())
                mensaje = await serializar_mensaje(int(datetime.utcnow().timestamp() % 1000000), self.canal, profit, True)
                await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
                self.logger.info(f"Pool actualizado con profit: {profit}, total: {self.pool}")
        except Exception as e:
            self.logger.error(f"Error actualizando pool: {e}")
            self.circuit_breaker.register_failure()

    async def assign_capital(self) -> None:
        if not self.circuit_breaker.check():
            return
        try:
            phase = await self.get_current_phase()
            risk_per_trade = phase["risk_per_trade"]
            max_active = self.pool * self.max_active_ratio
            risk_adjustment = 1.0
            if self.macro_context.get("vix_price", 0) > 20:
                risk_adjustment = 0.5
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
                mensaje = await serializar_mensaje(int(datetime.utcnow().timestamp() % 1000000), self.canal, trade_amount, True)
                await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
                self.active_capital += trade_amount
                await self.plugin_db.update_active_capital(self.active_capital)
                self.logger.info(f"Asignado {trade_amount} para trading, fase: {phase['name']} 🌟")
        except Exception as e:
            self.logger.error(f"Error asignando capital: {e}")
            self.circuit_breaker.register_failure()

    async def manage_pool(self):
        while True:
            await self.assign_capital()
            await asyncio.sleep(300)

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            if mensaje.get("tipo") == "trading_results":
                await self.update_pool_from_results(mensaje)
            elif mensaje.get("tipo") == "macro_data":
                self.macro_context = mensaje
                self.logger.info(f"Datos macro recibidos: {self.macro_context}")
        except Exception as e:
            self.logger.error(f"Error procesando evento: {e}")
            self.circuit_breaker.register_failure()

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("CapitalProcessor detenido")