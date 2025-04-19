#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/execution_processor.py
Gestiona la ejecución de órdenes de trading y backtesting avanzado con Bollinger Bands.
"""
from corec.core import ComponenteBase, zstd, serializar_mensaje
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker
import aiohttp
import time
import hmac
import hashlib
import backoff
import numpy as np
import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta

class ExecutionProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("ExecutionProcessor")
        self.plugin_db = TradingDB(self.config.get("db_config", {}))
        self.circuit_breakers = {
            ex["name"]: CircuitBreaker(
                self.config.get("execution_config", {}).get("circuit_breaker", {}).get("max_failures", 3),
                self.config.get("execution_config", {}).get("circuit_breaker", {}).get("reset_timeout", 900)
            ) for ex in self.config.get("exchange_config", {}).get("exchanges", [])
        }
        self.risk_per_trade = self.config.get("execution_config", {}).get("risk_per_trade", 0.02)
        self.take_profit = self.config.get("execution_config", {}).get("take_profit", 0.05)
        self.stop_loss = self.config.get("execution_config", {}).get("stop_loss", 0.02)

    async def inicializar(self):
        await self.plugin_db.connect()
        self.logger.info("ExecutionProcessor inicializado")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def place_order(self, exchange: Dict[str, Any], symbol: str, side: str, quantity: float, market: str, price: float) -> Dict[str, Any]:
        if not self.circuit_breakers[exchange["name"]].check():
            return {}
        try:
            name = exchange["name"]
            headers = {}
            params = {}
            url = ""
            timestamp = str(int(time.time() * 1000))
            if name == "binance":
                params = {
                    "symbol": symbol.replace('/', ''),
                    "side": side.upper(),
                    "type": "LIMIT",
                    "quantity": f"{quantity:.8f}",
                    "price": f"{price:.2f}",
                    "timestamp": timestamp
                }
                query = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
                signature = hmac.new(exchange["api_secret"].encode(), query.encode(), hashlib.sha256).hexdigest()
                params["signature"] = signature
                headers["X-MBX-APIKEY"] = exchange["api_key"]
                url = f"https://api.binance.com/api/v3/order" if market == "spot" else f"https://fapi.binance.com/fapi/v1/order"
            elif name == "kucoin":
                params = {
                    "symbol": symbol.replace('/', '-'),
                    "side": side.lower(),
                    "type": "limit",
                    "quantity": quantity,
                    "price": price,
                    "clientOid": f"order_{timestamp}"
                }
                headers["KC-API-KEY"] = exchange["api_key"]
                headers["KC-API-SIGN"] = hmac.new(exchange["api_secret"].encode(), json.dumps(params).encode(), hashlib.sha256).hexdigest()
                url = f"https://api.kucoin.com/api/v1/orders"
            elif name == "bybit":
                params = {
                    "symbol": symbol.replace('/', ''),
                    "side": side.capitalize(),
                    "order_type": "Limit",
                    "qty": quantity,
                    "price": price,
                    "time_in_force": "GTC",
                    "timestamp": timestamp
                }
                query = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
                signature = hmac.new(exchange["api_secret"].encode(), query.encode(), hashlib.sha256).hexdigest()
                headers["api_key"] = exchange["api_key"]
                headers["sign"] = signature
                url = f"https://api.bybit.com/v2/private/order/create"
            elif name == "okx":
                params = {
                    "instId": symbol.replace('/', '-'),
                    "tdMode": "cash",
                    "side": side.lower(),
                    "ordType": "limit",
                    "sz": quantity,
                    "px": price,
                    "clOrdId": f"order_{timestamp}"
                }
                headers["OK-ACCESS-KEY"] = exchange["api_key"]
                headers["OK-ACCESS-SIGN"] = hmac.new(exchange["api_secret"].encode(), json.dumps(params).encode(), hashlib.sha256).hexdigest()
                url = f"https://www.okx.com/api/v5/trade/order"
            elif name == "kraken":
                params = {
                    "pair": symbol.replace('/', ''),
                    "type": side.lower(),
                    "ordertype": "limit",
                    "volume": quantity,
                    "price": price,
                    "nonce": timestamp
                }
                headers["API-Key"] = exchange["api_key"]
                headers["API-Sign"] = hmac.new(exchange["api_secret"].encode(), "&".join([f"{k}={v}" for k, v in params.items()]).encode(), hashlib.sha256).hexdigest()
                url = f"https://futures.kraken.com/api/v3/order"
            async with aiohttp.ClientSession() as session:
                method = "POST"
                async with session.request(method, url, headers=headers, json=params if name in ["kucoin", "bybit", "okx"] else None, data=params if name not in ["kucoin", "bybit", "okx"] else None) as resp:
                    if resp.status == 200:
                        order = await resp.json()
                        order_id = order.get("orderId", order.get("id", order.get("data", {}).get("orderId", "unknown")))
                        order_data = {
                            "exchange": name,
                            "order_id": order_id,
                            "symbol": symbol,
                            "market": market,
                            "side": side,
                            "quantity": quantity,
                            "price": price,
                            "status": "open",
                            "timestamp": datetime.utcnow().timestamp()
                        }
                        await self.plugin_db.save_order(**order_data)
                        datos_comprimidos = zstd.compress(json.dumps(order_data).encode())
                        mensaje = await serializar_mensaje(int(order_data["timestamp"] % 1000000), self.canal, order_data["price"], True)
                        await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
                        self.logger.info(f"Orden colocada en {name}: {order_id} 🌟")
                        return order_data
                    else:
                        error = await resp.text()
                        self.logger.error(f"Error colocando orden en {name}: {resp.status} - {error}")
                        self.circuit_breakers[name].register_failure()
                        await self.nucleus.publicar_alerta({
                            "tipo": "api_error",
                            "plugin": "crypto_trading",
                            "exchange": name,
                            "message": f"Error {resp.status}: {error}"
                        })
                        return {}
        except Exception as e:
            self.logger.error(f"Error colocando orden en {name}: {e}")
            self.circuit_breakers[name].register_failure()
            await self.nucleus.publicar_alerta({
                "tipo": "api_exception",
                "plugin": "crypto_trading",
                "exchange": name,
                "message": str(e)
            })
            return {}

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def close_order(self, exchange_name: str, order: Dict[str, Any], reason: str) -> None:
        if not self.circuit_breakers[exchange_name].check():
            return
        try:
            exchange = next(ex for ex in self.config["exchange_config"]["exchanges"] if ex["name"] == exchange_name)
            headers = {}
            params = {}
            url = ""
            timestamp = str(int(time.time() * 1000))
            if exchange_name == "binance":
                params = {
                    "symbol": order["symbol"].replace('/', ''),
                    "orderId": order["order_id"],
                    "timestamp": timestamp
                }
                query = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
                signature = hmac.new(exchange["api_secret"].encode(), query.encode(), hashlib.sha256).hexdigest()
                params["signature"] = signature
                headers["X-MBX-APIKEY"] = exchange["api_key"]
                url = f"https://api.binance.com/api/v3/order" if order["market"] == "spot" else f"https://fapi.binance.com/fapi/v1/order"
            elif exchange_name == "kucoin":
                params = {"orderId": order["order_id"]}
                headers["KC-API-KEY"] = exchange["api_key"]
                headers["KC-API-SIGN"] = hmac.new(exchange["api_secret"].encode(), json.dumps(params).encode(), hashlib.sha256).hexdigest()
                url = f"https://api.kucoin.com/api/v1/orders/{order['order_id']}"
            elif exchange_name == "bybit":
                params = {"symbol": order["symbol"].replace('/', ''), "orderId": order["order_id"], "timestamp": timestamp}
                query = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
                signature = hmac.new(exchange["api_secret"].encode(), query.encode(), hashlib.sha256).hexdigest()
                headers["api_key"] = exchange["api_key"]
                headers["sign"] = signature
                url = f"https://api.bybit.com/v2/private/order/cancel"
            elif exchange_name == "okx":
                params = {"instId": order["symbol"].replace('/', '-'), "ordId": order["order_id"], "clOrdId": f"order_{timestamp}"}
                headers["OK-ACCESS-KEY"] = exchange["api_key"]
                headers["OK-ACCESS-SIGN"] = hmac.new(exchange["api_secret"].encode(), json.dumps(params).encode(), hashlib.sha256).hexdigest()
                url = f"https://www.okx.com/api/v5/trade/cancel-order"
            elif exchange_name == "kraken":
                params = {"orderids": order["order_id"], "nonce": timestamp}
                headers["API-Key"] = exchange["api_key"]
                headers["API-Sign"] = hmac.new(exchange["api_secret"].encode(), "&".join([f"{k}={v}" for k, v in params.items()]).encode(), hashlib.sha256).hexdigest()
                url = f"https://futures.kraken.com/api/v3/cancelorder"
            async with aiohttp.ClientSession() as session:
                method = "POST"
                async with session.request(method, url, headers=headers, json=params if exchange_name in ["kucoin", "bybit", "okx"] else None, data=params if exchange_name not in ["kucoin", "bybit", "okx"] else None) as resp:
                    if resp.status == 200:
                        order["status"] = "closed"
                        order["close_reason"] = reason
                        order["close_timestamp"] = datetime.utcnow().timestamp()
                        await self.plugin_db.save_order(**order)
                        datos_comprimidos = zstd.compress(json.dumps(order).encode())
                        mensaje = await serializar_mensaje(int(order["close_timestamp"] % 1000000), self.canal, 0.0, False)
                        await self.redis_client.xadd("crypto_trading_data", {"data": mensaje})
                        self.logger.info(f"Orden cerrada en {exchange_name}: {order['order_id']} por {reason}")
                    else:
                        error = await resp.text()
                        self.logger.error(f"Error cerrando orden en {exchange_name}: {resp.status} - {error}")
                        self.circuit_breakers[exchange_name].register_failure()
                        await self.nucleus.publicar_alerta({
                            "tipo": "api_error",
                            "plugin": "crypto_trading",
                            "exchange": exchange_name,
                            "message": f"Error {resp.status}: {error}"
                        })
        except Exception as e:
            self.logger.error(f"Error cerrando orden en {exchange_name}: {e}")
            self.circuit_breakers[exchange_name].register_failure()
            await self.nucleus.publicar_alerta({
                "tipo": "api_exception",
                "plugin": "crypto_trading",
                "exchange": exchange_name,
                "message": str(e)
            })

    async def calculate_bollinger_bands(self, prices: List[float], window: int = 20, num_std: float = 2.0) -> tuple:
        if len(prices) < window:
            return [], [], []
        prices = np.array(prices)
        sma = np.convolve(prices, np.ones(window)/window, mode='valid')
        std = np.array([np.std(prices[i:i+window]) for i in range(len(prices)-window+1)])
        upper_band = sma + num_std * std
        lower_band = sma - num_std * std
        return sma.tolist(), upper_band.tolist(), lower_band.tolist()

    async def run_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            risk = params.get("risk", self.risk_per_trade)
            take_profit = params.get("take_profit", self.take_profit)
            stop_loss = params.get("stop_loss", self.stop_loss)
            symbol = params.get("symbol", "BTC/USDT")
            trades = params.get("trades", 20)
            conn = psycopg2.connect(**self.config["db_config"])
            cur = conn.cursor()
            cur.execute(
                "SELECT price, timestamp FROM market_data WHERE symbol = %s ORDER BY timestamp DESC LIMIT 1000",
                (symbol,)
            )
            prices = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            if len(prices) < 20:
                return {"estado": "error", "mensaje": "Datos insuficientes para backtest"}
            sma, upper_band, lower_band = await self.calculate_bollinger_bands(prices)
            capital = 1000.0
            position = 0.0
            entry_price = 0.0
            total_trades = 0
            profits = []
            for i in range(20, len(prices)):
                if total_trades >= trades:
                    break
                price = prices[i]
                if i >= len(sma):
                    continue
                bb_upper = upper_band[i-20]
                bb_lower = lower_band[i-20]
                rsi = np.random.uniform(30, 70)  # Simulado
                if price < bb_lower and rsi < 30 and position == 0:
                    position = (capital * risk) / price
                    entry_price = price
                    capital -= position * price
                    total_trades += 1
                elif position > 0:
                    if price >= entry_price * (1 + take_profit) or price > bb_upper:
                        profit = position * (price - entry_price)
                        capital += position * price
                        profits.append(profit)
                        position = 0
                    elif price <= entry_price * (1 - stop_loss) or price < bb_lower:
                        loss = position * (entry_price - price)
                        capital += position * price
                        profits.append(-loss)
                        position = 0
            roi = (sum(profits) / 1000.0) * 100 if profits else 0
            sharpe_ratio = (np.mean(profits) / np.std(profits)) * np.sqrt(252) if profits and np.std(profits) != 0 else 0
            return {
                "estado": "ok",
                "roi": roi,
                "trades": total_trades,
                "sharpe_ratio": sharpe_ratio,
                "profits": profits
            }
        except Exception as e:
            self.logger.error(f"Error en backtest: {e}")
            return {"estado": "error", "mensaje": str(e)}

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            if mensaje.get("tipo") == "predictor_temporal":
                symbol = mensaje.get("symbol")
                prediction = mensaje.get("prediction")
                if symbol and prediction:
                    price_data = self.metrics_cache.get("market_data", {}).get(symbol, {})
                    current_price = price_data.get("price")
                    if current_price:
                        exchange = self.config["exchange_config"]["exchanges"][0]
                        quantity = (1000 * self.risk_per_trade) / current_price
                        if prediction > current_price * (1 + self.take_profit):
                            await self.place_order(exchange, symbol, "buy", quantity, "spot", current_price)
                        elif prediction < current_price * (1 - self.stop_loss):
                            await self.place_order(exchange, symbol, "sell", quantity, "spot", current_price)
            elif mensaje.get("tipo") == "trading_execution" and mensaje.get("action") == "update_risk":
                self.risk_per_trade = mensaje["risk_per_trade"]
                self.logger.info(f"Riesgo actualizado a {self.risk_per_trade}")
            elif mensaje.get("tipo") == "trading_execution" and mensaje.get("action") == "update_strategy":
                self.risk_per_trade = mensaje["params"]["risk_per_trade"]
                self.take_profit = mensaje["params"]["take_profit"]
                self.logger.info(f"Estrategia actualizada: riesgo {self.risk_per_trade}, take-profit {self.take_profit}")
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
        self.logger.info("ExecutionProcessor detenido")