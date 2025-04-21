import logging
import json
import random
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker
from typing import Dict, Any  # Añadimos las importaciones necesarias

class PredictorProcessor:
    def __init__(self, config, redis):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("PredictorProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 900)
        )

    async def inicializar(self):
        """Inicializa el predictor."""
        self.logger.info("PredictorProcessor inicializado")

    async def predecir_tendencias(self) -> Dict[str, Any]:
        """Predice tendencias de mercado basadas en datos históricos."""
        if not self.cb.check():
            self.logger.warning("Circuit breaker activo, omitiendo predicción de tendencias")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            market_data = await self.redis.get("market_data")
            if not market_data:
                self.logger.warning("No hay datos de mercado disponibles para predecir tendencias")
                return {"status": "error", "motivo": "no_data"}

            market_data = json.loads(market_data)
            crypto_data = market_data["crypto"]

            # Simulación simple de predicción de tendencias
            trend = "alcista" if random.random() > 0.5 else "bajista"
            magnitude = random.uniform(0.01, 0.05)

            result = {
                "trend": trend,
                "magnitude": magnitude,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.redis.set("predicted_trend", json.dumps(result))
            self.logger.info(f"Tendencia predicha: {result}")
            return {"status": "ok", "data": result}
        except Exception as e:
            self.logger.error("Error al predecir tendencias", exc_info=True)
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}

    async def adjust_trading_flow(self):
        """Ajusta el flujo de trading basándose en predicciones."""
        while True:
            try:
                trend_data = await self.redis.get("predicted_trend")
                if not trend_data:
                    self.logger.warning("No hay datos de tendencia disponibles para ajustar el flujo")
                    await asyncio.sleep(60)
                    continue

                trend_data = json.loads(trend_data)
                trend = trend_data.get("trend", "lateral")
                magnitude = trend_data.get("magnitude", 0.01)

                adjustments = {
                    "interval_factor": 1.0,
                    "trade_multiplier": 2,
                    "pause": False
                }

                if trend == "alcista" and magnitude > 0.03:
                    adjustments["interval_factor"] = 0.8  # Reducir intervalo para operar más rápido
                    adjustments["trade_multiplier"] = 3    # Aumentar número de operaciones
                elif trend == "bajista" and magnitude > 0.03:
                    adjustments["pause"] = True  # Pausar operaciones en mercados bajistas fuertes
                else:
                    adjustments["interval_factor"] = 1.0
                    adjustments["trade_multiplier"] = 2

                # Simular distracciones humanas (5% de probabilidad de omitir un ciclo)
                if random.random() < 0.05:
                    self.logger.info("Simulando distracción humana, omitiendo ajuste de flujo")
                    await asyncio.sleep(60)
                    continue

                await self.redis.set("trading_flow_adjustments", json.dumps(adjustments))
                self.logger.info(f"Ajustes de flujo aplicados: {adjustments}")
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"Error al ajustar flujo de trading: {e}")
                await asyncio.sleep(60)
