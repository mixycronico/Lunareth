import logging
from typing import Dict, List
from datetime import datetime
from plugins.crypto_trading.utils.settlement_utils import calcular_ganancias, registrar_historial

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
        self.daily_profit_loss = 0

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

    async def daily_close_loop(self, open_trades: Dict[str, Dict], close_trade_callback):
        """Cierra todas las operaciones abiertas a las 10:00 PM y genera un resumen diario."""
        while True:
            try:
                now = datetime.datetime.now()
                close_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
                if now.hour == 22 and now.minute == 0:
                    daily_profit = 0
                    for trade_id, trade in list(open_trades.items()):
                        exchange, pair = trade_id.split(":")
                        current_price = 50000
                        profit = trade["cantidad"] * (current_price - trade["precio"])
                        daily_profit += profit
                        await registrar_historial(self.redis_client, f"trade_history:{exchange}:{pair}", {
                            "precio_entrada": trade["precio"],
                            "precio_salida": current_price,
                            "cantidad": trade["cantidad"],
                            "timestamp": trade["close_timestamp"]
                        })
                        await close_trade_callback(exchange, pair, trade)
                    self.daily_profit_loss += daily_profit
                    await self.nucleus.publicar_alerta({
                        "tipo": "cierre_diario",
                        "plugin_id": "crypto_trading",
                        "profit_loss": daily_profit,
                        "total_profit_loss": self.daily_profit_loss,
                        "capital": self.total_capital,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    await self.trading_db.save_report(
                        date=now.strftime("%Y-%m-%d"),
                        total_profit=daily_profit,
                        roi_percent=(daily_profit / self.total_capital) * 100 if self.total_capital != 0 else 0,
                        total_trades=len(open_trades),
                        report_data={"open_trades": 0},
                        timestamp=now.timestamp()
                    )
                    self.logger.info(f"[CapitalProcessor] Cierre diario: Ganancia/Pérdida: ${daily_profit}, Total: ${self.daily_profit_loss}, Capital: ${self.total_capital}")
                    await asyncio.sleep(86400 - 60)
                else:
                    await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"[CapitalProcessor] Error en cierre diario: {e}")
                await asyncio.sleep(60)
