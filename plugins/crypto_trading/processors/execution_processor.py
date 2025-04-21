import logging
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker

class ExecutionProcessor:
    def __init__(self, config, redis):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("ExecutionProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 900)
        )

    async def ejecutar_operacion(self, exchange, orden: dict) -> dict:
        if not self.cb.check():
            self.logger.warning(f"[{exchange}] Circuito de ejecución activo, operación omitida")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            orden_id = f"{exchange}_{datetime.utcnow().timestamp()}"
            resultado = {
                "orden_id": orden_id,
                "status": "ejecutado",
                "precio": orden["precio"],
                "cantidad": orden["cantidad"],
                "activo": orden["activo"],
                "tipo": orden["tipo"],
                "timestamp": datetime.utcnow().isoformat()
            }
            self.logger.info(f"Orden ejecutada: {resultado}")
            return resultado
        except Exception as e:
            self.logger.exception(f"Error al ejecutar en {exchange}")
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}
