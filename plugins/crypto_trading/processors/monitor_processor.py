#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/monitor_processor.py
Monitorea precios en tiempo real desde crypto_trading_data y publica en market_data.
Consume altcoins desde macro_data y pondera precios por volumen de trading.
"""

from corec.core import ComponenteBase, zstd, serializar_mensaje
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker
import json
import asyncio
import logging
from typing import Dict, Any
from datetime import datetime


class MonitorProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("MonitorProcessor")
        monitor_cfg = self.config.get("monitor_config", {})
        self.symbols = monitor_cfg.get("symbols", [])
        self.update_interval = monitor_cfg.get("update_interval", 60)
        breaker_cfg = monitor_cfg.get("circuit_breaker", {})
        self.circuit_breaker = CircuitBreaker(
            breaker_cfg.get("max_failures", 3),
            breaker_cfg.get("reset_timeout", 900)
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
            all_symbols = self.symbols + self.altcoins
            for symbol in all_symbols:
                if symbol in self.price_cache:
                    entry = self.price_cache[symbol]
                    if now < entry["expires"]:
                        vols = [v["volume"] for v in entry["data"].values()]
                        prices = [
                            v["price"] * v["volume"]
                            for v in entry["data"].values()
                        ]
                        total_volume = sum(vols)
                        if total_volume > 0:
                            weighted_price = sum(prices) / total_volume
                            price_data = {
                                "symbol": symbol,
                                "price": weighted_price,
                                "timestamp": now
                            }
                            mensaje = await serializar_mensaje(
                                int(now % 1000000),
                                self.canal,
                                weighted_price,
                                True
                            )
                            await self.redis_client.xadd(
                                "market_data", {"data": mensaje}
                            )
                            await self.plugin_db.save_price(
                                exchange="weighted",
                                symbol=symbol,
                                market="spot",
                                price=weighted_price,
                                timestamp=now
                            )
                            self.logger.debug(
                                f"Precio ponderado {symbol}: {weighted_price} 🌟"
                            )
                    else:
                        del self.price_cache[symbol]
            await asyncio.sleep(self.update_interval)

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            if mensaje.get("tipo") == "macro_data" and "altcoins" in mensaje:
                self.altcoins = mensaje.get("altcoins", [])
                self.logger.info(f"Altcoins actualizados: {self.altcoins}")
            elif mensaje.get("tipo") == "exchange_data":
                symbol = mensaje["symbol"]
                exchange = mensaje["exchange"]
                if symbol not in self.price_cache:
                    self.price_cache[symbol] = {
                        "data": {},
                        "expires": datetime.utcnow().timestamp() + 300
                    }
                self.price_cache[symbol]["data"][exchange] = {
                    "price": mensaje["price"],
                    "volume": mensaje.get("volume", 0),
                    "timestamp": mensaje["timestamp"]
                }
                self.logger.debug(
                    f"{symbol} desde {exchange}: "
                    f"{mensaje['price']} vol {mensaje.get('volume', 0)}"
                )
        except Exception as e:
            self.logger.error(f"Error en evento: {e}")
            self.circuit_breaker.register_failure()

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("MonitorProcessor detenido")
