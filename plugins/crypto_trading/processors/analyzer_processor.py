import logging
import json
import numpy as np
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker
from typing import Dict, Any, List  # Añadimos las importaciones necesarias

class AnalyzerProcessor:
    def __init__(self, config, redis):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("AnalyzerProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 900)
        )
        self.min_volumen = 1000000
        self.min_cap = 100000000

    def _calcular_tendencias(self, volumenes: Dict[str, float], market_caps: Dict[str, float]) -> Dict[str, str]:
        """Calcula tendencias para cada símbolo."""
        tendencias = {}
        for symbol in volumenes:
            vol = volumenes[symbol]
            cap = market_caps[symbol]
            if vol > self.min_volumen and cap > self.min_cap:
                tendencia = "alcista" if vol / cap > 0.001 else "bajista"
            else:
                tendencia = "lateral"
            tendencias[symbol] = tendencia
        return tendencias

    def _top_altcoins(self, volumenes: Dict[str, float]) -> List[str]:
        altcoins = [k for k in volumenes if k not in ["BTC", "ETH"]]
        return sorted(altcoins, key=lambda x: volumenes[x], reverse=True)[:10]

    async def analizar_volatilidad(self, exchange: str, pairs: List[str]) -> Dict[str, Any]:
        """Analiza la volatilidad de los símbolos y prioriza pares para operar."""
        if not self.cb.check():
            self.logger.warning("Circuit breaker activo, omitiendo análisis de volatilidad")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            market_data = await self.redis.get("market_data")
            if not market_data:
                self.logger.warning("No hay datos de mercado disponibles para el análisis de volatilidad")
                return {"status": "error", "motivo": "no_data"}

            market_data = json.loads(market_data)
            crypto_data = market_data["crypto"]

            resultados = []
            for symbol, data in crypto_data.items():
                if symbol not in pairs:
                    continue
                prices = [50000 + i * 100 for i in range(50)]
                if len(prices) < 2:
                    continue
                price_changes = np.diff(prices) / prices[:-1]
                volatility = np.std(price_changes) if len(price_changes) > 0 else 0.01
                is_volatile = volatility >= self.config.get("volatility_threshold", 0.025)
                resultados.append({
                    "symbol": symbol,
                    "volatilidad": volatility,
                    "alerta": is_volatile,
                    "timestamp": datetime.utcnow().isoformat()
                })

            # Priorizar pares
            prioritized_pairs = []
            for res in resultados:
                if res["alerta"]:
                    prioritized_pairs.append((res["symbol"], res["volatilidad"]))
            for symbol in pairs:
                if not any(p[0] == symbol for p in prioritized_pairs):
                    prioritized_pairs.append((symbol, 0.01))

            result = {
                "exchange": exchange,
                "prioritized_pairs": prioritized_pairs,
                "avg_volatility": sum(v for _, v in prioritized_pairs) / len(prioritized_pairs) if prioritized_pairs else 0.01,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.redis.set(f"volatility_data:{exchange}", json.dumps(result))
            self.logger.info(f"Volatilidad analizada para {exchange}: {len(prioritized_pairs)} pares priorizados")
            return {"status": "ok", "datos": result}
        except Exception as e:
            self.logger.error("Error en análisis de volatilidad", exc_info=True)
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}

    async def analizar(self) -> Dict[str, Any]:
        """Analiza volúmenes y tendencias."""
        if not self.cb.check():
            self.logger.warning("Circuit breaker activo, omitiendo análisis")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            volumenes = {
                "BTC": 1000000,
                "ETH": 500000,
                "ALT1": 200000,
                "ALT2": 150000,
                "ALT3": 120000,
                "ALT4": 110000
            }
            market_caps = {
                "BTC": 1000000000,
                "ETH": 500000000,
                "ALT1": 200000000,
                "ALT2": 150000000,
                "ALT3": 120000000,
                "ALT4": 110000000
            }
            tendencias = self._calcular_tendencias(volumenes, market_caps)
            top_altcoins = self._top_altcoins(volumenes)

            result = {
                "tendencias": tendencias,
                "top_altcoins": top_altcoins,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.redis.set("analyzer_data", json.dumps(result))
            self.logger.info(f"Análisis completado: {result}")
            return {"status": "ok", "data": result}
        except Exception as e:
            self.logger.error("Error al analizar datos", exc_info=True)
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}
