import logging
from typing import Dict, List
from datetime import datetime
from plugins.crypto_trading.utils.settlement_utils import calcular_ganancias, registrar_historial
import random

class CapitalProcessor:
    def __init__(self, config, redis_client, trading_db, nucleus):
        self.logger = logging.getLogger("CapitalProcessor")
        self.config = config
        self.redis_client = redis_client
        self.trading_db = trading_db
        self.nucleus = nucleus
        self.total_capital = config.get("total_capital", 1000.0)
        self.allocated = {}
        self.percentage_to_use = 0.7
        self.min_per_trade = 0.01
        self.max_per_trade = 0.20
        self.base_capital = 100
        self.strategy_performance = {"momentum": 0.0, "scalping": 0.0}
        self.strategy_weights = {"momentum": 0.5, "scalping": 0.5}  # Proporciones iniciales

    async def update_strategy_performance(self, daily_profit: float):
        self.strategy_performance["momentum"] += daily_profit / self.base_capital * self.strategy_weights["momentum"]
        self.strategy_performance["scalping"] += daily_profit / self.base_capital * self.strategy_weights["scalping"]
        self.logger.info(f"Rendimiento actualizado - Momentum: {self.strategy_performance['momentum']}, Scalping: {self.strategy_performance['scalping']}")

    async def vote_strategy(self):
        """Ajusta las proporciones de Momentum y Scalping cada 24 horas según rendimiento."""
        while True:
            try:
                momentum_score = self.strategy_performance["momentum"]
                scalping_score = self.strategy_performance["scalping"]
                total_score = momentum_score + scalping_score
                if total_score == 0:
                    self.strategy_weights = {"momentum": 0.5, "scalping": 0.5}
                else:
                    self.strategy_weights["momentum"] = momentum_score / total_score
                    self.strategy_weights["scalping"] = scalping_score / total_score
                self.logger.info(f"Proporciones ajustadas - Momentum: {self.strategy_weights['momentum']*100:.2f}%, Scalping: {self.strategy_weights['scalping']*100:.2f}%")
                await self.redis_client.set("trading_strategy", json.dumps(self.strategy_weights))
                await asyncio.sleep(86400)
            except Exception as e:
                self.logger.error(f"Error al votar estrategia: {e}")
                await asyncio.sleep(86400)

    async def adjust_base_capital(self):
        while True:
            try:
                performance = self.total_capital / self.base_capital - 1
                if performance > 0.5:
                    self.base_capital = min(self.base_capital * 1.2, 1000)
                elif performance < -0.2:
                    self.base_capital = max(self.base_capital * 0.8, 50)
                self.logger.info(f"Capital base ajustado por consenso: ${self.base_capital}")
                await asyncio.sleep(86400)
            except Exception as e:
                self.logger.error(f"Error al ajustar el capital base: {e}")
                await asyncio.sleep(86400)

    def distribuir_capital(self, exchanges_disponibles: List[str]) -> Dict[str, float]:
        self.logger.info("Distribuyendo capital entre exchanges disponibles...")
        capital_utilizable = self.total_capital * self.percentage_to_use
        num_ex = len(exchanges_disponibles)

        if num_ex == 0:
            self.logger.warning("No hay exchanges disponibles para asignar capital.")
            return {}

        por_exchange = capital_utilizable / num_ex
        self.allocated = {ex: por_exchange for ex in exchanges_disponibles}
        self.logger.info(f"Capital distribuido: {self.allocated}")
        return self.allocated

    def calcular_monto_operacion(self, exchange: str, confianza: float) -> float:
        capital = self.allocated.get(exchange, 0.0)
        if capital == 0.0:
            self.logger.warning(f"No hay capital asignado a {exchange}")
            return 0.0

        porcentaje = max(self.min_per_trade, min(confianza, self.max_per_trade))
        monto = capital * porcentaje
        self.logger.debug(f"Monto sugerido para operar en {exchange}: ${monto:.2f} (confianza: {confianza})")
        return monto

    def actualizar_total_capital(self, nuevo_total: float):
        self.logger.info(f"Actualizando capital total: {self.total_capital} → {nuevo_total}")
        self.total_capital = nuevo_total
