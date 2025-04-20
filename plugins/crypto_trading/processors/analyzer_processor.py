#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/analyzer_processor.py
Analiza volúmenes y tendencias para BTC, ETH y altcoins principales.
"""

import logging
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker

class AnalyzerProcessor:
    def __init__(self, config, redis):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("AnalyzerProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 900)
        )

    async def analizar(self, precios: dict, volumenes: dict) -> dict:
        if not self.cb.check():
            self.logger.warning("Análisis omitido por circuito caído")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            activos = ["BTC", "ETH"] + self._top_altcoins(volumenes)
            resultados = {}
            for activo in activos:
                tendencia = self._calcular_tendencia(precios[activo])
                resultados[activo] = {
                    "tendencia": tendencia,
                    "timestamp": datetime.utcnow().isoformat()
                }
            return {"status": "ok", "data": resultados}
        except Exception as e:
            self.logger.exception("Error en analizar()")
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}

    def _top_altcoins(self, volumenes: dict) -> list:
        altcoins = [k for k in volumenes if k not in ["BTC", "ETH"]]
        return sorted(altcoins, key=lambda x: volumenes[x], reverse=True)[:10]

    def _calcular_tendencia(self, precios: list) -> str:
        if len(precios) < 2:
            return "estable"
        delta = precios[-1] - precios[0]
        if delta > 0:
            return "alcista"
        elif delta < 0:
            return "bajista"
        return "lateral"