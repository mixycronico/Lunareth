#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/monitor_processor.py
Monitorea precios en tiempo real de criptomonedas desde crypto_trading_data y publica en market_data.
Consume altcoins dinámicos desde macro_data. Pondera precios por volumen de trading.
"""
from corec.core import ComponenteBase, zstd, serializar_mensaje
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker
import json
import asyncio
import numpy as np
from typing import Dict, Any, List
from datetime import datetime, timedelta

class MonitorProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("MonitorProcessor")
        self.symbols = self.config.get("monitor_config", {}).get("symbols", [])
        self.update_interval = self.config.get("monitor_config", {}).get("update_interval", 60)
        self.circuit_breaker = CircuitBreaker(
            self.config.get("monitor_config", {}).get("circuit_breaker", {}).get("max_failures", 3),
            self.config.get("monitor_config", {}).get("circuit_breaker", {}).get("reset_timeout", 900)
        )
        self.plugin_db = TradingDB(self.config.get("db_config", {}))
        self.altcoins = []
        self.price_cache = {}

    async def inicializar(self):
        await self.plugin_db.connect()
        asyncio.create_task(self.monitor_prices())
        self.logger.info("MonitorProcessor inicializado")

    async def monitor_prices(self):
        while True:
            if not self.circuit_breaker.check():
                await asyncio.sleep(60)
                continue
            now = datetime.utcnow().timestamp()
            for symbol in self.symbols + self.altcoins:
                if symbol in self.price_cache:
                    cache_entry = self.price_cache[symbol]
                    if now < cache_entry["expires"]:
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
                        datos_comprimidos = zstd.compress(json.dumps(price_data).encode())
                        mensaje = await serializar_mensaje(int(now % 1000000), self.canal, weighted_price, True)
                        await self.redis_client.xadd("market_data", {"data": mensaje})
                        await self.plugin_db.save_price(
                            exchange="weighted",
                            symbol=symbol,
                            market="spot",
                            price=weighted_price,
                            timestamp=now
                        )
                        self.logger.debug(f"Precio ponderado para {symbol}: {weighted_price} 🌟")
                    else:
                        del self.price_cache[symbol]
            await asyncio.sleep(self.update_interval)

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            if mensaje.get("tipo") == "macro_data" and "altcoins" in mensaje:
                self.altcoins = mensaje.get("altcoins", [])
                self.logger.info(f"Altcoins actualizados: {self.altcoins}")
            elif mensaje.get("tipo") == "exchange_data":
                price_data = mensaje
                symbol = price_data["symbol"]
                exchange = price_data["exchange"]
                if symbol not in self.price_cache:
                    self.price_cache[symbol] = {
                        "data": {},
                        "expires": datetime.utcnow().timestamp() + 300
                    }
                self.price_cache[symbol]["data"][exchange] = {
                    "price": price_data["price"],
                    "volume": price_data.get("volume", 0),
                    "timestamp": price_data["timestamp"]
                }
                self.logger.debug(f"Precio recibido para {symbol} desde {exchange}: {price_data['price']}, volumen: {price_data.get('volume', 0)}")
        except Exception as e:
            self.logger.error(f"Error procesando evento: {e}")
            self.circuit_breaker.register_failure()

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("MonitorProcessor detenido")