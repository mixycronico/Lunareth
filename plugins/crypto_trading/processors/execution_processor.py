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
        self.capital = config.get("capital", 100)
        self.active_capital = 0  # Capital actualmente invertido

    def get_phase(self) -> int:
        if self.capital < 1000:
            return 1
        elif self.capital < 10000:
            return 2
        elif self.capital < 100000:
            return 3
        elif self.capital < 1000000:
            return 4
        elif self.capital < 10000000:
            return 5
        else:
            return 6

    def get_execution_params(self) -> Dict[str, float]:
        phase = self.get_phase()
        phase_params = {
            1: {"risk_per_trade": 0.02, "take_profit": 0.03, "stop_loss": 0.01},
            2: {"risk_per_trade": 0.018, "take_profit": 0.03, "stop_loss": 0.01},
            3: {"risk_per_trade": 0.013, "take_profit": 0.04, "stop_loss": 0.015},
            4: {"risk_per_trade": 0.007, "take_profit": 0.04, "stop_loss": 0.015},
            5: {"risk_per_trade": 0.003, "take_profit": 0.05, "stop_loss": 0.02},
            6: {"risk_per_trade": 0.001, "take_profit": 0.05, "stop_loss": 0.02}
        }
        return phase_params[phase]

    def can_trade(self, exchange: str, num_exchanges: int) -> bool:
        """Verifica si se puede operar según el límite del 70% del capital."""
        max_capital_per_exchange = (self.capital * 0.7) / num_exchanges
        exchange_active_capital = sum(
            trade["cantidad"] for trade_id, trade in self.config.get("open_trades", {}).items()
            if trade_id.startswith(exchange + ":")
        )
        return exchange_active_capital < max_capital_per_exchange

    async def ejecutar_operacion(self, exchange, orden: dict, paper_mode: bool = False, trade_multiplier: int = 1) -> dict:
        if not self.cb.check():
            self.logger.warning(f"[{exchange}] Circuito de ejecución activo, operación omitida")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            params = self.get_execution_params()
            risk_per_trade = params["risk_per_trade"]
            take_profit = params["take_profit"]
            stop_loss = params["stop_loss"]

            # Multiplicar operaciones durante subidas
            for _ in range(trade_multiplier):
                if not self.can_trade(exchange, self.config.get("num_exchanges", 2)):
                    self.logger.warning(f"[{exchange}] Límite de capital alcanzado para este exchange")
                    break

                if paper_mode:
                    orden_id = f"paper_{exchange}_{datetime.utcnow().timestamp()}"
                    resultado = {
                        "orden_id": orden_id,
                        "status": "ejecutado",
                        "precio": orden["precio"],
                        "cantidad": orden["cantidad"] * risk_per_trade,
                        "activo": orden["activo"],
                        "tipo": orden["tipo"],
                        "timestamp": datetime.utcnow().isoformat(),
                        "modo": "paper",
                        "take_profit": orden["precio"] * (1 + take_profit),
                        "stop_loss": orden["precio"] * (1 - stop_loss)
                    }
                    self.active_capital += resultado["cantidad"]
                    self.logger.info(f"Orden simulada (paper mode): {resultado}")
                else:
                    orden_id = f"{exchange}_{datetime.utcnow().timestamp()}"
                    resultado = {
                        "orden_id": orden_id,
                        "status": "ejecutado",
                        "precio": orden["precio"],
                        "cantidad": orden["cantidad"] * risk_per_trade,
                        "activo": orden["activo"],
                        "tipo": orden["tipo"],
                        "timestamp": datetime.utcnow().isoformat(),
                        "modo": "real",
                        "take_profit": orden["precio"] * (1 + take_profit),
                        "stop_loss": orden["precio"] * (1 - stop_loss)
                    }
                    self.active_capital += resultado["cantidad"]
                    self.logger.info(f"Orden ejecutada: {resultado}")
                yield resultado

        except Exception as e:
            self.logger.exception(f"Error al ejecutar en {exchange}")
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}

    def update_capital(self, new_capital: float):
        self.capital = new_capital
        self.logger.info(f"[ExecutionProcessor] Capital actualizado: ${self.capital}, Fase: {self.get_phase()}")

    def reset_active_capital(self):
        """Resetea el capital activo después del cierre diario."""
        self.active_capital = 0
        self.logger.info("[ExecutionProcessor] Capital activo reseteado")
