import logging
import json
import pickle
import os
import random  # Añadimos la importación de random que se usa en adjust_trading_flow
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker
from plugins.crypto_trading.utils.settlement_utils import calcular_ganancias, registrar_historial
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
import asyncpg
import asyncio  # Añadimos la importación de asyncio que se usa explícitamente
from typing import Dict, Any  # Añadimos las importaciones necesarias

class RNNPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=1):
        super(RNNPredictor, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class PredictorProcessor:
    def __init__(self, config, redis):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("PredictorProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 900)
        )
        self.window = 10
        self.history = []
        self.distraction_probability = 0.05  # 5% de probabilidad de "distracción"

    async def predecir_tendencias(self) -> Dict[str, Any]:
        if not self.cb.check():
            self.logger.warning("Circuit breaker activo, omitiendo predicción de tendencias")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            market_data = await self.redis.get("market_data")
            if not market_data:
                self.logger.warning("No hay datos de mercado disponibles para la predicción")
                return {"status": "error", "motivo": "no_data"}

            market_data = json.loads(market_data)
            crypto_data = market_data["crypto"]

            avg_price_change = sum(
                (data["price_change"] if "price_change" in data else 0) for data in crypto_data.values()
            ) / len(crypto_data)
            
            self.history.append(avg_price_change)
            if len(self.history) > self.window:
                self.history.pop(0)

            trend = "alcista" if avg_price_change > 0.01 else "bajista" if avg_price_change < -0.01 else "neutral"
            trend_strength = abs(avg_price_change)

            result = {
                "predicciones": [{"symbol": symbol, "tendencia": trend} for symbol in crypto_data.keys()],
                "trend_strength": trend_strength,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.redis.set("market_trends", json.dumps(result))
            self.logger.info(f"Tendencias predichas: {result}")
            return {"status": "ok", "predicciones": result["predicciones"]}
        except Exception as e:
            self.logger.error("Error al predecir tendencias", exc_info=True)
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}

    async def adjust_trading_flow(self) -> None:
        """Ajusta dinámicamente el flujo de operaciones cada 5 minutos, con pausas y distracciones."""
        while True:
            try:
                # Simular distracción humana (omitir ciclo con 5% de probabilidad si el mercado está estable)
                market_trends = await self.redis.get("market_trends")
                if market_trends:
                    trends = json.loads(market_trends)
                    trend_strength = trends.get("trend_strength", 0.0)
                    if trend_strength < 0.01 and random.random() < self.distraction_probability:
                        self.logger.info("Simulando distracción humana: omitiendo ciclo de ajuste")
                        await asyncio.sleep(300)
                        continue

                # Pausa aleatoria durante baja volatilidad
                if trend_strength < 0.01:
                    pause = random.uniform(300, 600)  # Pausa de 5-10 minutos
                    self.logger.info(f"Simulando pausa humana durante baja volatilidad: {pause} segundos")
                    await asyncio.sleep(pause)

                market_trends = await self.redis.get("market_trends")
                if not market_trends:
                    await asyncio.sleep(300)
                    continue

                trends = json.loads(market_trends)
                trend_strength = trends.get("trend_strength", 0.0)
                trend = "alcista" if trend_strength > 0 else "bajista"

                adjustment = {
                    "interval_factor": 0.8 if trend == "alcista" else 1.2,
                    "trade_multiplier": 4 if trend == "alcista" else 2,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await self.redis.set("trading_flow_adjustments", json.dumps(adjustment))
                self.logger.info(f"Flujo de trading ajustado: {adjustment}")
            except Exception as e:
                self.logger.error(f"Error al ajustar el flujo de trading: {e}")
            await asyncio.sleep(300)
