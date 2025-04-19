#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/exchange_processor.py
Consulta precios y órdenes abiertas en 5 exchanges, publicando datos en crypto_trading_data.
"""
from corec.core import ComponenteBase, zstd, serializar_mensaje
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker
import aiohttp
import asyncio
import json
import hmac
import hashlib
import backoff
from typing import Dict, Any, List
from datetime import datetime, timedelta

class ExchangeProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("ExchangeProcessor")
        self.exchanges = self.config.get("exchange_config", {}).get("exchanges", [])
        self.fetch_interval = self.config.get("exchange_config", {}).get("fetch_interval", 300) / len(self.exchanges)
        self.circuit_breakers = {
            ex["name"]: CircuitBreaker(
                self.config.get("exchange_config", {}).get("circuit_breaker", {}).get("max_failures", 3),
                self.config.get("exchange_config", {}).get("circuit_breaker", {}).get("reset_timeout", 900)
            ) for ex in self.exchanges
        }
        self.plugin_db = TradingDB(self.config.get("db_config", {}))

    async def inicializar(self):
        await self.plugin_db.connect()
        for exchange in self.exchanges:
            asyncio.create_task(self.fetch_exchange_data(exchange))
        self.logger.info("ExchangeProcessor inicializado")

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
                    self.circuit_breakers[name].register_failure()
                    return {}
        except Exception as e:
            self.logger.error(f"Error obteniendo precio spot para {symbol} en {name}: {e}")
            self.circuit_breakers[name].register_failure()
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
                    self.circuit_breakers[name].register_failure()
                    return {}
        except Exception as e:
            self.logger.error(f"Error obteniendo precio futures para {symbol} en {name}: {e}")
            self.circuit_breakers[name].register_failure()
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
                url = f"https://futures.kraken.com/api/v3/openorders"

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
                    self.circuit_breakers[name].register_failure()
                    return []
        except Exception as e:
            self.logger.error(f"Error obteniendo órdenes abiertas en {name}: {e}")
            self.circuit_breakers[name].register_failure()
            return []

    async def fetch_exchange_data(self, exchange: Dict[str, Any]):
        while True:
            if not self.circuit_breakers[exchange["name"]].check():
                await asyncio.sleep(60)
                continue
            async with aiohttp.ClientSession() as session:
                for symbol in exchange["symbols"]:
                    spot_data = await self.fetch_spot_price(exchange, symbol, session)
                    futures_data = await self.fetch_futures_price(exchange, symbol, session)
                    for price_data in [spot_data, futures_data]:
                        if price_data:
                            await self.redis_client.setex(
                                f"price:{exchange['name']}:{symbol}:{price_data['market']}",
                                1800,
                                json.dumps(price_data)
                            )
                            datos_comprimidos = zstd.compress(json.dumps(price_data).encode())
                            mensaje = await serializar_mensaje(
                                int(price_data["timestamp"] % 1000000), self.canal, price_data["price"], True
                            )
                            await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
                            await self.plugin_db.save_price(**price_data)
                            self.logger.debug(f"Precio {price_data['market']} {symbol} en {exchange['name']}: {price_data['price']}")
                open_orders = await self.fetch_open_orders(exchange, session)
                for order in open_orders:
                    datos_comprimidos = zstd.compress(json.dumps(order).encode())
                    mensaje = await serializar_mensaje(
                        int(order["timestamp"] % 1000000), self.canal, 0.0, order["status"] == "open"
                    )
                    await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
                    await self.plugin_db.save_order(**order)
                    self.logger.debug(f"Orden abierta {order['order_id']} en {exchange['name']}: {order['status']}")
            await asyncio.sleep(self.fetch_interval)

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("ExchangeProcessor detenido")