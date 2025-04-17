#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/market_monitor/processors/monitor_processor.py
"""
monitor_processor.py
Monitorea precios en tiempo real de criptomonedas desde exchange_data y publica en market_data.
Consume altcoins dinámicos desde macro_data. Pondera precios por volumen de trading.
"""

from ....core.processors.base import ProcesadorBase
from ....core.entidad_base import Event
from ....utils.logging import logger
from ..utils.db import MonitorDB
import json
import zstandard as zstd
from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio

class MonitorProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("MonitorProcessor")
        self.symbols = self.config.get("config", {}).get("symbols", [])
        self.update_interval = self.config.get("config", {}).get("update_interval", 60)
        self.circuit_breaker = self.config.get("config", {}).get("circuit_breaker", {})
        self.plugin_db = None
        self.failure_count = 0
        self.breaker_tripped = False
        self.breaker_reset_time = None
        self.altcoins = []  # Lista dinámica de altcoins desde macro_data
        self.price_cache = {}  # Caché local para precios con TTL y volumen

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        # Inicializar base de datos
        self.plugin_db = MonitorDB(self.db_config)
        if not await self.plugin_db.connect():
            self.logger.warning("No se pudo conectar a monitor_db, usando almacenamiento temporal")

        # Iniciar monitoreo
        asyncio.create_task(self.monitor_prices())
        self.logger.info("MonitorProcessor inicializado")

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
            await self.nucleus.publicar_alerta({"tipo": "circuit_breaker_tripped", "plugin": "market_monitor"})

    async def monitor_prices(self):
        while True:
            if not await self.check_circuit_breaker():
                await asyncio.sleep(60)
                continue

            now = datetime.utcnow().timestamp()
            for symbol in self.symbols + self.altcoins:
                if symbol in self.price_cache:
                    cache_entry = self.price_cache[symbol]
                    if now < cache_entry["expires"]:
                        # Ponderar precios por volumen
                        total_volume = sum(entry["volume"] for entry in cache_entry["data"].values())
                        weighted_price = 0.0
                        if total_volume > 0:
                            weighted_price = sum(
                                entry["price"] * entry["volume"] for entry in cache_entry["data"].values()
                            ) / total_volume
                        price_data = {
                            "symbol": symbol,
                            "price": weighted_price,
                            "timestamp": now
                        }
                        # Publicar en Redis
                        datos_comprimidos = zstd.compress(json.dumps(price_data).encode())
                        await self.redis_client.xadd("market_data", {"data": datos_comprimidos})
                        # Guardar en base de datos
                        if self.plugin_db and self.plugin_db.conn:
                            await self.plugin_db.save_price(
                                symbol=price_data["symbol"],
                                price=price_data["price"],
                                timestamp=price_data["timestamp"]
                            )
                        self.logger.debug(f"Precio ponderado para {symbol}: {weighted_price}")
                    else:
                        del self.price_cache[symbol]  # Expirar entrada
            await asyncio.sleep(self.update_interval)

    async def manejar_evento(self, event: Event) -> None:
        if event.canal == "macro_data" and "altcoins" in event.datos:
            self.altcoins = event.datos.get("altcoins", [])
            self.logger.info(f"Altcoins actualizados: {self.altcoins}")
        elif event.canal == "exchange_data":
            try:
                price_data = json.loads(zstd.decompress(event.datos["data"]))
                symbol = price_data["symbol"]
                exchange = price_data["exchange"]
                if symbol not in self.price_cache:
                    self.price_cache[symbol] = {
                        "data": {},
                        "expires": datetime.utcnow().timestamp() + 300  # TTL de 300 segundos
                    }
                self.price_cache[symbol]["data"][exchange] = {
                    "price": price_data["price"],
                    "volume": price_data.get("volume", 0),  # Volumen de trading
                    "timestamp": price_data["timestamp"]
                }
                self.logger.debug(f"Precio recibido para {symbol} desde {exchange}: {price_data['price']}, volumen: {price_data.get('volume', 0)}")
            except Exception as e:
                self.logger.error(f"Error procesando precio: {e}")
                await self.register_failure()

    async def detener(self):
        if self.plugin_db:
            await self.plugin_db.disconnect()
        self.logger.info("MonitorProcessor detenido")