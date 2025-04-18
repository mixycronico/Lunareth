#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/execution_processor.py
Ejecuta órdenes basadas en señales del predictor o estrategias.
"""

import logging
from typing import Dict, Any
from corec.core import ComponenteBase
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker


class ExecutionProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("ExecutionProcessor")
        self.plugin_db = TradingDB(self.config.get("db_config", {}))

        self.circuit_breakers = {
            ex["name"]: CircuitBreaker(
                self.config.get("execution_config", {})
                .get("circuit_breaker", {}).get("max_failures", 3),
                self.config.get("execution_config", {})
                .get("circuit_breaker", {}).get("reset_timeout", 900)
            ) for ex in self.config.get("exchange_config", {})
            .get("exchanges", [])
        }

        exe_cfg = self.config.get("execution_config", {})
        self.risk_per_trade = exe_cfg.get("risk_per_trade", 0.02)
        self.take_profit = exe_cfg.get("take_profit", 0.05)
        self.stop_loss = exe_cfg.get("stop_loss", 0.02)
        self.metrics_cache = {}

    async def inicializar(self):
        await self.plugin_db.connect()
        self.logger.info("ExecutionProcessor inicializado")

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            tipo = mensaje.get("tipo")
            if tipo == "predictor_temporal":
                symbol = mensaje.get("symbol")
                prediction = mensaje.get("prediction")

                if symbol and prediction:
                    price_data = self.metrics_cache.get(
                        "market_data", {}
                    ).get(symbol, {})
                    current_price = price_data.get("price")

                    if not current_price:
                        raise ValueError(
                            f"No hay precio actual para {symbol} en cache"
                        )

                    exchange = self.config["exchange_config"]["exchanges"][0]
                    quantity = (1000 * self.risk_per_trade) / current_price

                    if prediction > current_price * (1 + self.take_profit):
                        await self.place_order(
                            exchange, symbol, "buy",
                            quantity, "spot", current_price
                        )
                    elif prediction < current_price * (1 - self.stop_loss):
                        await self.place_order(
                            exchange, symbol, "sell",
                            quantity, "spot", current_price
                        )

            elif tipo == "trading_execution":
                action = mensaje.get("action")
                if action == "update_risk":
                    self.risk_per_trade = mensaje["risk_per_trade"]
                    self.logger.info(
                        f"Riesgo actualizado a {self.risk_per_trade}"
                    )
                elif action == "update_strategy":
                    params = mensaje["params"]
                    self.risk_per_trade = params["risk_per_trade"]
                    self.take_profit = params["take_profit"]
                    self.logger.info(
                        f"Estrategia actualizada: riesgo {self.risk_per_trade}, "
                        f"take-profit {self.take_profit}"
                    )

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
