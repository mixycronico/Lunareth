#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/macro_processor.py
Análisis macroeconómico para decisiones estratégicas.
"""

import logging
import aiohttp
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker

class MacroProcessor:
    def __init__(self, config, redis):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("MacroProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 900)
        )
        self.sources = config.get("macro_sources", [
            "https://api.coindesk.com/v1/bpi/currentprice.json"
        ])

    async def obtener_macro_indicadores(self):
        if not self.cb.check():
            self.logger.warning("Circuit breaker activo, omitiendo análisis macroeconómico")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            resultados = []
            async with aiohttp.ClientSession() as session:
                for url in self.sources:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            resultados.append({
                                "fuente": url,
                                "datos": data,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        else:
                            self.logger.warning(f"Fallo en fuente macro: {url} [{response.status}]")

            self.logger.info(f"Indicadores macro recopilados: {len(resultados)} fuentes")
            return {"status": "ok", "fuentes": resultados}
        except Exception as e:
            self.logger.exception("Error durante análisis macroeconómico")
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}