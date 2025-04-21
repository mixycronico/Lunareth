import logging
from typing import Dict, List

class CapitalProcessor:
    def __init__(self, config):
        self.logger = logging.getLogger("CapitalProcessor")
        self.config = config
        self.total_capital = config.get("total_capital", 1000.0)
        self.allocated = {}
        self.percentage_to_use = 0.7  # Usar el 70% del capital
        self.min_per_trade = 0.01
        self.max_per_trade = 0.20

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
