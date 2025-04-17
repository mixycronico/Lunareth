#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/exchange_sync/processors/exchange_processor.py
"""
exchange_processor.py
Consulta precios y operaciones abiertas en 5 exchanges, publicando datos en exchange_data.
"""

from ....core.processors.base import ProcesadorBase
from ....utils.logging import logger
from ..utils.db import ExchangeDB
import aiohttp
import asyncio
import json
import zstandard as zstd
from typing import Dict, Any, List
from datetime import datetime, timedelta
import backoff
import hmac
import hashlib

class ExchangeProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("ExchangeProcessor")
        self.exchanges = config.get("exchange_config", {}).get("exchanges", [])
        self.fetch_interval = config.get("config", {}).get("fetch_interval", 300) / len(self.exchanges)
        self.circuit_breakers = {ex["name"]: {"tripped": False, "failure_count": 0, "reset_time": None} for ex in self.exchanges}
        self.plugin_db = None

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        # Inicializar base de datos
        self.plugin_db = ExchangeDB(self.db_config)
        if not await self.plugin_db.connect():
            self.logger.warning("No se pudo conectar a exchange_db, usando almacenamiento temporal")

        # Iniciar tareas por exchange
        for exchange in self.exchanges:
            asyncio.create_task(self.fetch_exchange_data(exchange))
        self.logger.info("ExchangeProcessor inicializado")

    async def check_circuit_breaker(self, exchange_name: str) -> bool:
        breaker = self.circuit_breakers[exchange_name]
        if breaker["tripped"]:
            now = datetime.utcnow()
            if now >= breaker["reset_time"]:
                breaker["tripped"] = False
                breaker["failure_count"] = 0
                breaker["reset_time"] = None
                self.logger.info(f"Circuit breaker reseteado para {exchange_name}")
            else:
                self.logger.warning(f"Circuit breaker activo para {exchange_name} hasta {breaker['reset_time']}")
                return False
        return True

    async def register_failure(self, exchange_name: str) -> None:
        breaker = self.circuit_breakers[exchange_name]
        breaker["failure_count"] += 1
        if breaker["failure_count"] >= self.config["config"]["circuit_breaker"]["max_failures"]:
            breaker["tripped"] = True
            breaker["reset_time"] = datetime.utcnow() + timedelta(seconds=self.config["config"]["circuit_breaker"]["reset_timeout"])
            self.logger.error(f"Circuit breaker activado para {exchange_name} hasta {breaker['reset_time']}")
            await self.nucleus.publicar_alerta({"tipo": "circuit_breaker_tripped", "plugin": "exchange_sync", "exchange": exchange_name})

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def fetch_spot_price(self, exchange: Dict[str, Any], symbol: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            name = exchange["name"]
            headers = {}
            url = ""

            if name == "binance":
                url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol.replace('/', '')}"
            elif name == "kucoin":
                url = f"https://api.kucoin.com/api/v1/market/orderbook/level1?symbol={symbol.replace('/', '-')}"
            elif name == "bybit":
                url = f"https://api.bybit.com/v2/public/tickers?symbol={symbol.replace('/', '')}"
            elif name == "okx":
                url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol.replace('/', '-')}"
            elif name == "kraken":
                url = f"https://api.kraken.com/0/public/Ticker?pair={symbol.replace('/', '')}"

            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = 0.0
                    if name == "binance":
                        price = float(data.get("price", 0))
                    elif name == "kucoin":
                        price = float(data.get("data", {}).get("price", 0))
                    elif name == "bybit":
                        price = float(data.get("result", [{}])[0].get("last_price", 0))
                    elif name == "okx":
                        price = float(data.get("data", [{}])[0].get("last", 0))
                    elif name == "kraken":
                        price = float(data.get("result", {}).get(symbol.replace('/', ''), {}).get("c", [0])[0])

                    return {
                        "exchange": name,
                        "symbol": symbol,
                        "market": "spot",
                        "price": price,
                        "timestamp": datetime.utcnow().timestamp()
                    }
                else:
                    self.logger.error(f"Error obteniendo precio spot para {symbol} en {name}: {resp.status}")
                    await self.register_failure(name)
                    return {}
        except Exception as e:
            self.logger.error(f"Error obteniendo precio spot para {symbol} en {name}: {e}")
            await self.register_failure(name)
            return {}

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def fetch_futures_price(self, exchange: Dict[str, Any], symbol: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        try:
            name = exchange["name"]
            headers = {}
            url = ""

            if name == "binance":
                url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol.replace('/', '')}"
            elif name == "kucoin":
                url = f"https://api-futures.kucoin.com/api/v1/ticker?symbol={symbol.replace('/', '-')}"
            elif name == "bybit":
                url = f"https://api.bybit.com/v2/public/tickers?symbol={symbol.replace('/', '')}"
            elif name == "okx":
                url = f"https://www.okx.com/api/v5/market/ticker?instId={symbol.replace('/', '-')}-FUT"
            elif name == "kraken":
                url = f"https://futures.kraken.com/api/v3/ticker?symbol={symbol.replace('/', '')}"

            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    price = 0.0
                    if name == "binance":
                        price = float(data.get("price", 0))
                    elif name == "kucoin":
                        price = float(data.get("data", {}).get("price", 0))
                    elif name == "bybit":
                        price = float(data.get("result", [{}])[0].get("last_price", 0))
                    elif name == "okx":
                        price = float(data.get("data", [{}])[0].get("last", 0))
                    elif name == "kraken":
                        price = float(data.get("tickers", [{}])[0].get("last", 0))

                    return {
                        "exchange": name,
                        "symbol": symbol,
                        "market": "futures",
                        "price": price,
                        "timestamp": datetime.utcnow().timestamp()
                    }
                else:
                    self.logger.error(f"Error obteniendo precio futures para {symbol} en {name}: {resp.status}")
                    await self.register_failure(name)
                    return {}
        except Exception as e:
            self.logger.error(f"Error obteniendo precio futures para {symbol} en {name}: {e}")
            await self.register_failure(name)
            return {}

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def fetch_open_orders(self, exchange: Dict[str, Any], session: aiohttp.ClientSession) -> List[Dict[str, Any]]:
        try:
            name = exchange["name"]
            headers = {}
            url = ""
            params = {"timestamp": int(datetime.utcnow().timestamp() * 1000)}

            if name == "binance":
                signature = hmac.new(exchange["api_secret"].encode(), "&".join([f"{k}={v}" for k, v in params.items()]).encode(), hashlib.sha256).hexdigest()
                params["signature"] = signature
                headers["X-MBX-APIKEY"] = exchange["api_key"]
                url = f"https://api.binance.com/api/v3/openOrders"
            elif name == "kucoin":
                signature = hmac.new(exchange["api_secret"].encode(), "&".join([f"{k}={v}" for k, v in params.items()]).encode(), hashlib.sha256).hexdigest()
                headers["KC-API-KEY"] = exchange["api_key"]
                headers["KC-API-SIGN"] = signature
                url = f"https://api.kucoin.com/api/v1/orders"
            elif name == "bybit":
                signature = hmac.new(exchange["api_secret"].encode(), "&".join([f"{k}={v}" for k, v in params.items()]).encode(), hashlib.sha256).hexdigest()
                headers["api_key"] = exchange["api_key"]
                headers["sign"] = signature
                url = f"https://api.bybit.com/v2/private/order/list"
            elif name == "okx":
                signature = hmac.new(exchange["api_secret"].encode(), "&".join([f"{k}={v}" for k, v in params.items()]).encode(), hashlib.sha256).hexdigest()
                headers["OK-ACCESS-KEY"] = exchange["api_key"]
                headers["OK-ACCESS-SIGN"] = signature
                url = f"https://www.okx.com/api/v5/trade/orders-pending"
            elif name == "kraken":
                signature = hmac.new(exchange["api_secret"].encode(), "&".join([f"{k}={v}" for k, v in params.items()]).encode(), hashlib.sha256).hexdigest()
                headers["API-Key"] = exchange["api_key"]
                headers["API-Sign"] = signature
                url = f"https://api.kraken.com/0/private/OpenOrders"

            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    orders = await resp.json()
                    return [
                        {
                            "exchange": name,
                            "order_id": order.get("orderId", order.get("id", "")),
                            "symbol": order.get("symbol"),
                            "market": "spot" if order.get("type", "").lower() == "spot" else "futures",
                            "status": order.get("status", "open"),
                            "timestamp": datetime.utcnow().timestamp()
                        } for order in orders.get("data", orders) if isinstance(orders, list) or "data" in orders
                    ]
                else:
                    self.logger.error(f"Error obteniendo órdenes abiertas en {name}: {resp.status}")
                    await self.register_failure(name)
                    return []
        except Exception as e:
            self.logger.error(f"Error obteniendo órdenes abiertas en {name}: {e}")
            await self.register_failure(name)
            return []

    async def fetch_exchange_data(self, exchange: Dict[str, Any]):
        while True:
            if not await self.check_circuit_breaker(exchange["name"]):
                await asyncio.sleep(60)
                continue

            async with aiohttp.ClientSession() as session:
                # Consultar precios (spot y futures)
                for symbol in exchange["symbols"]:
                    spot_data = await self.fetch_spot_price(exchange, symbol, session)
                    futures_data = await self.fetch_futures_price(exchange, symbol, session)
                    for price_data in [spot_data, futures_data]:
                        if price_data:
                            # Cachear en Redis
                            await self.redis_client.setex(
                                f"price:{exchange['name']}:{symbol}:{price_data['market']}",
                                1800,
                                json.dumps(price_data)
                            )
                            # Publicar en Redis
                            datos_comprimidos = zstd.compress(json.dumps(price_data).encode())
                            await self.redis_client.xadd("exchange_data", {"data": datos_comprimidos})
                            # Guardar en base de datos
                            if self.plugin_db and self.plugin_db.conn:
                                await self.plugin_db.save_price(
                                    exchange=price_data["exchange"],
                                    symbol=price_data["symbol"],
                                    market=price_data["market"],
                                    price=price_data["price"],
                                    timestamp=price_data["timestamp"]
                                )
                            self.logger.debug(f"Precio {price_data['market']} {symbol} en {exchange['name']}: {price_data['price']}")

                # Consultar órdenes abiertas
                open_orders = await self.fetch_open_orders(exchange, session)
                for order in open_orders:
                    datos_comprimidos = zstd.compress(json.dumps(order).encode())
                    await self.redis_client.xadd("exchange_data", {"data": datos_comprimidos})
                    if self.plugin_db and self.plugin_db.conn:
                        await self.plugin_db.save_order(
                            exchange=order["exchange"],
                            order_id=order["order_id"],
                            symbol=order["symbol"],
                            market=order["market"],
                            status=order["status"],
                            timestamp=order["timestamp"]
                        )
                    self.logger.debug(f"Orden abierta {order['order_id']} en {exchange['name']}: {order['status']}")

            await asyncio.sleep(self.fetch_interval)

    async def detener(self):
        if self.plugin_db:
            await self.plugin_db.disconnect()
        self.logger.info("ExchangeProcessor detenido")