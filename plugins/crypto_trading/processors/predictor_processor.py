#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/predictor_processor.py
Predicción de tendencias usando modelos AI ligeros entrenados con datos recientes.
"""

import logging
import aiohttp
import numpy as np
from datetime import datetime, timedelta
from plugins.crypto_trading.utils.helpers import CircuitBreaker

class PredictorProcessor:
    def __init__(self, config, redis):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("PredictorProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 600)
        )
        self.symbols = config.get("symbols", ["BTC/USDT", "ETH/USDT"])
        self.history_limit = config.get("history_limit", 96)  # ~24h si es cada 15 min

    async def obtener_historial(self, symbol):
        key = f"market:history:{symbol.replace('/', '')}"
        datos = await self.redis.lrange(key, -self.history_limit, -1)
        return [float(x) for x in datos] if datos else []

    def evaluar_tendencia(self, precios: list) -> str:
        if len(precios) < 2:
            return "indefinida"
        x = np.arange(len(precios))
        y = np.array(precios)
        slope = np.polyfit(x, y, 1)[0]
        if slope > 0.02:
            return "alcista"
        elif slope < -0.02:
            return "bajista"
        else:
            return "neutral"

    async def predecir_tendencias(self):
        if not self.cb.check():
            self.logger.warning("Circuit breaker activo, omitiendo predicción")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        resultados = []
        try:
            for symbol in self.symbols:
                historial = await self.obtener_historial(symbol)
                tendencia = self.evaluar_tendencia(historial)
                resultados.append({
                    "symbol": symbol,
                    "tendencia": tendencia,
                    "timestamp": datetime.utcnow().isoformat()
                })

            self.logger.info("Predicción completada")
            return {"status": "ok", "predicciones": resultados}
        except Exception as e:
            self.logger.error("Error al predecir tendencias", exc_info=True)
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}