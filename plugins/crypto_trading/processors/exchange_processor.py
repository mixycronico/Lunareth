#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/exchange_processor.py
Consulta precios y órdenes abiertas en exchanges, publica en crypto_trading_data.
"""

import aiohttp
import asyncio
import logging
import backoff
from typing import Dict, Any, List
from datetime import datetime

from corec.core import ComponenteBase
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker


class ExchangeProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("ExchangeProcessor")
        self.exchanges = self.config.get("exchange_config", {}).get("exchanges", [])
        interval = self.config.get("exchange_config", {}).get("fetch_interval", 300)
        self.fetch_interval = interval / max(len(self.exchanges), 1)
        self.circuit_breakers = {
            ex["name"]: CircuitBreaker(
                self.config.get("exchange_config", {}).get("circuit_breaker", {}).get(
                    "max_failures", 3
                ),
                self.config.get("exchange_config", {}).get("circuit_breaker", {}).get(
                    "reset_timeout", 900
                )
            ) for ex in self.exchanges
        }
        self.plugin_db = TradingDB(self.config.get("db_config", {}))

    async def inicializar(self):
        await self.plugin_db.connect()
        for exchange in self.exchanges:
            asyncio.create_task(self.fetch_exchange_data(exchange))
        self.logger.info("ExchangeProcessor inicializado")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def fetch_spot_price(
        self, exchange: Dict[str, Any], symbol: str,
        session: aiohttp.ClientSession
    ) -> Dict[str, Any]:
        try:
            name = exchange["name"]
            headers = {}
            url = ""

            sym_noslash = symbol.replace('/', '')
            sym_dash = symbol.replace('/', '-')

            if name == "binance":
                url = (
                    "https://api.binance.com/api/v3/ticker/price"
                    f"?symbol={sym_noslash}"
                )
            elif name == "kucoin":
                url = (
                    "https://api.kucoin.com/api/v1/market/"
                    f"orderbook/level1?symbol={sym_dash}"
                )
            elif name == "bybit":
                url = (
                    "https://api.bybit.com/v2/public/"
                    f"tickers?symbol={sym_noslash}"
                )
            elif name == "okx":
                url = (
                    "https://www.okx.com/api/v5/market/"
                    f"ticker?instId={sym_dash}"
                )
            elif name == "kraken":
                url = (
                    "https://api.kraken.com/0/public/Ticker"
                    f"?pair={sym_noslash}"
                )

            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = 0.0
                    if name == "binance":
                        price = float(data.get("price", 0))
                    elif name == "kucoin":
                        price = float(data.get("data", {}).get("price", 0))
                    elif name == "bybit":
                        price = float(
                            data.get("result", [{}])[0].get("last_price", 0)
                        )
                    elif name == "okx":
                        price = float(data.get("data", [{}])[0].get("last", 0))
                    elif name == "kraken":
                        price = float(
                            data.get("result", {}).get(sym_noslash, {})
                            .get("c", [0])[0]
                        )
                    return {
                        "exchange": name,
                        "symbol": symbol,
                        "market": "spot",
                        "price": price,
                        "timestamp": datetime.utcnow().timestamp()
                    }
                else:
                    self.logger.error(
                        f"Error spot {symbol} en {name}: {resp.status}"
                    )
                    self.circuit_breakers[name].register_failure()
                    return {}
        except Exception as e:
            self.logger.error(f"Error spot {symbol} en {name}: {e}")
            self.circuit_breakers[name].register_failure()
            return {}

    async def fetch_exchange_data(self, exchange: Dict[str, Any]):
        while True:
            await asyncio.sleep(self.fetch_interval)

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("ExchangeProcessor detenido")
