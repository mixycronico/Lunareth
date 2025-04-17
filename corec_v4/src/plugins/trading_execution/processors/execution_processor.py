#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/trading_execution/processors/execution_processor.py
"""
execution_processor.py
Gestiona la ejecución de órdenes de trading y backtesting avanzado con Bollinger Bands.
"""

from ....core.processors.base import ProcesadorBase
from ....core.entidad_base import Event
from ....utils.logging import logger
from ..utils.db import ExecutionDB
import aiohttp
import time
import hmac
import hashlib
import zstandard as zstd
import json
from typing import Dict, Any
from datetime import datetime, timedelta
import backoff
import psycopg2
import numpy as np

class ExecutionProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("ExecutionProcessor")
        self.plugin_db = None
        self.circuit_breaker = config.get("config", {}).get("circuit_breaker", {})
        self.failure_count = {ex["name"]: 0 for ex in config["exchange_config"]["exchanges"]}
        self.breaker_tripped = {ex["name"]: False for ex in config["exchange_config"]["exchanges"]}
        self.breaker_reset_time = {ex["name"]: None for ex in config["exchange_config"]["exchanges"]}
        self.risk_per_trade = config.get("execution_config", {}).get("risk_per_trade", 0.02)
        self.take_profit = config.get("execution_config", {}).get("take_profit", 0.05)
        self.stop_loss = config.get("execution_config", {}).get("stop_loss", 0.02)

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.plugin_db = ExecutionDB(self.db_config)
        if not await self.plugin_db.connect():
            self.logger.warning("No se pudo conectar a execution_db")
            await self.nucleus.publicar_alerta({"tipo": "db_connection_error", "plugin": "trading_execution", "message": "No se pudo conectar a execution_db"})
        self.logger.info("ExecutionProcessor inicializado")

    async def check_circuit_breaker(self, exchange_name: str) -> bool:
        if self.breaker_tripped[exchange_name]:
            now = datetime.utcnow()
            if now >= self.breaker_reset_time[exchange_name]:
                self.breaker_tripped[exchange_name] = False
                self.failure_count[exchange_name] = 0
                self.breaker_reset_time[exchange_name] = None
                self.logger.info(f"Circuit breaker reseteado para {exchange_name}")
            else:
                self.logger.warning(f"Circuit breaker activo para {exchange_name} hasta {self.breaker_reset_time[exchange_name]}")
                return False
        return True

    async def register_failure(self, exchange_name: str) -> None:
        self.failure_count[exchange_name] += 1
        if self.failure_count[exchange_name] >= self.circuit_breaker.get("max_failures", 3):
            self.breaker_tripped[exchange_name] = True
            self.breaker_reset_time[exchange_name] = datetime.utcnow() + timedelta(seconds=self.circuit_breaker.get("reset_timeout", 900))
            self.logger.error(f"Circuit breaker activado para {exchange_name} hasta {self.breaker_reset_time[exchange_name]}")
            await self.nucleus.publicar_alerta({"tipo": "circuit_breaker_tripped", "plugin": "trading_execution", "exchange": exchange_name})

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def place_order(self, exchange: Dict[str, Any], symbol: str, side: str, quantity: float, market: str, price: float) -> Dict[str, Any]:
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
            # ... (resto de exchanges igual)

            async with aiohttp.ClientSession() as session:
                method = "POST" if name in ["kucoin", "bybit", "okx"] else "POST"
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
                        if self.plugin_db and self.plugin_db.conn:
                            await self.plugin_db.save_order(**order_data)
                        return order_data
                    else:
                        error = await resp.text()
                        self.logger.error(f"Error colocando orden en {name}: {resp.status} - {error}")
                        await self.register_failure(name)
                        await self.nucleus.publicar_alerta({"tipo": "api_error", "plugin": "trading_execution", "exchange": name, "message": f"Error {resp.status}: {error}"})
                        return {}
        except Exception as e:
            self.logger.error(f"Error colocando orden en {name}: {e}")
            await self.register_failure(name)
            await self.nucleus.publicar_alerta({"tipo": "api_exception", "plugin": "trading_execution", "exchange": name, "message": str(e)})
            return {}

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def close_order(self, exchange_name: str, order: Dict[str, Any], reason: str) -> None:
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
            # ... (resto de exchanges igual)

            async with aiohttp.ClientSession() as session:
                method = "POST" if exchange_name in ["kucoin", "bybit", "okx"] else "POST"
                async with session.request(method, url, headers=headers, json=params if exchange_name in ["kucoin", "bybit", "okx"] else None, data=params if name not in ["kucoin", "bybit", "okx"] else None) as resp:
                    if resp.status == 200:
                        order["status"] = "closed"
                        order["close_reason"] = reason
                        order["close_timestamp"] = datetime.utcnow().timestamp()
                        if self.plugin_db and self.plugin_db.conn:
                            await self.plugin_db.save_order(**order)
                        datos_comprimidos = zstd.compress(json.dumps(order).encode())
                        await self.redis_client.xadd("trading_results", {"data": datos_comprimidos})
                        self.logger.info(f"Orden cerrada en {exchange_name}: {order['order_id']} por {reason}")
                    else:
                        error = await resp.text()
                        self.logger.error(f"Error cerrando orden en {exchange_name}: {resp.status} - {error}")
                        await self.register_failure(exchange_name)
                        await self.nucleus.publicar_alerta({"tipo": "api_error", "plugin": "trading_execution", "exchange": exchange_name, "message": f"Error {resp.status}: {error}"})
        except Exception as e:
            self.logger.error(f"Error cerrando orden en {exchange_name}: {e}")
            await self.register_failure(exchange_name)
            await self.nucleus.publicar_alerta({"tipo": "api_exception", "plugin": "trading_execution", "exchange": exchange_name, "message": str(e)})

    async def calculate_bollinger_bands(self, prices: list, window: int = 20, num_std: float = 2.0) -> tuple:
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

            # Obtener datos históricos
            conn = psycopg2.connect(**self.db_config)
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

            # Calcular Bollinger Bands
            sma, upper_band, lower_band = await self.calculate_bollinger_bands(prices)

            # Simular operaciones
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
                "roi": roi,
                "trades": total_trades,
                "sharpe_ratio": sharpe_ratio,
                "profits": profits
            }
        except Exception as e:
            self.logger.error(f"Error en backtest: {e}")
            return {"estado": "error", "mensaje": str(e)}

    async def manejar_evento(self, event: Event) -> None:
        try:
            datos = json.loads(zstd.decompress(event.datos["data"]))
            if event.canal == "corec_stream_corec1":
                symbol = datos.get("symbol")
                prediction = datos.get("prediction")
                if symbol and prediction:
                    price_data = self.data_cache.get("market_data", {}).get(symbol, {})
                    current_price = price_data.get("price")
                    if current_price:
                        if prediction > current_price * (1 + self.take_profit):
                            exchange = self.config["exchange_config"]["exchanges"][0]
                            quantity = (1000 * self.risk_per_trade) / current_price
                            await self.place_order(exchange, symbol, "buy", quantity, "spot", current_price)
                        elif prediction < current_price * (1 - self.stop_loss):
                            exchange = self.config["exchange_config"]["exchanges"][0]
                            quantity = (1000 * self.risk_per_trade) / current_price
                            await self.place_order(exchange, symbol, "sell", quantity, "spot", current_price)
            elif event.canal == "trading_execution" and datos.get("action") == "update_risk":
                self.risk_per_trade = datos["risk_per_trade"]
                self.logger.info(f"Riesgo actualizado a {self.risk_per_trade}")
            elif event.canal == "trading_execution" and datos.get("action") == "update_strategy":
                self.risk_per_trade = datos["params"]["risk_per_trade"]
                self.take_profit = datos["params"]["take_profit"]
                self.logger.info(f"Estrategia actualizada: riesgo {self.risk_per_trade}, take-profit {self.take_profit}")
        except Exception as e:
            self.logger.error(f"Error manejando evento: {e}")
            await self.register_failure("event")
            await self.nucleus.publicar_alerta({"tipo": "event_error", "plugin": "trading_execution", "message": str(e)})

    async def detener(self):
        if self.plugin_db:
            await self.plugin_db.disconnect()
        self.logger.info("ExecutionProcessor detenido")