import logging
import json
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker
from plugins.crypto_trading.utils.settlement_utils import calcular_ganancias, registrar_historial

class SettlementProcessor:
    def __init__(self, config, redis, trading_db, nucleus, strategy, execution_processor, capital_processor):
        self.logger = logging.getLogger("SettlementProcessor")
        self.config = config
        self.redis = redis
        self.trading_db = trading_db
        self.nucleus = nucleus
        self.strategy = strategy
        self.execution_processor = execution_processor
        self.capital_processor = capital_processor
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 900)
        )
        self.capital = config.get("total_capital", 1000.0)
        self.daily_profit_loss = 0

    async def update_capital_after_trade(self, side: str, trade_result: Dict[str, Any]):
        """Actualiza el capital después de cada operación."""
        if side == "buy":
            self.capital -= trade_result["cantidad"]
        elif side == "sell":
            self.capital += trade_result["cantidad"]
        self.strategy.update_capital(self.capital)
        self.execution_processor.update_capital(self.capital)
        self.capital_processor.actualizar_total_capital(self.capital)
        self.logger.info(f"[SettlementProcessor] Capital actualizado: ${self.capital}")

    async def close_trade(self, exchange: str, pair: str, trade: dict, open_trades: Dict[str, Dict]):
        """Cierra una operación y actualiza el capital."""
        trade_id = f"{exchange}:{pair}"
        self.logger.info(f"[SettlementProcessor] Cerrando operación para {trade_id}")
        trade["status"] = "closed"
        trade["close_timestamp"] = datetime.datetime.utcnow().isoformat()
        await self.trading_db.save_order(
            exchange,
            trade["orden_id"],
            pair,
            "spot",
            "closed",
            datetime.datetime.utcnow().timestamp()
        )
        profit = trade["cantidad"] * (50000 - trade["precio"])
        self.capital += trade["cantidad"] + profit
        self.strategy.update_capital(self.capital)
        self.execution_processor.update_capital(self.capital)
        self.capital_processor.actualizar_total_capital(self.capital)
        del open_trades[trade_id]
        await self.nucleus.publicar_alerta({
            "tipo": "operacion_cerrada",
            "plugin_id": "crypto_trading",
            "exchange": exchange,
            "pair": pair,
            "timestamp": trade["close_timestamp"]
        })

    async def daily_close_loop(self, open_trades: Dict[str, Dict]):
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
                        await registrar_historial(self.redis, f"trade_history:{exchange}:{pair}", {
                            "precio_entrada": trade["precio"],
                            "precio_salida": current_price,
                            "cantidad": trade["cantidad"],
                            "timestamp": trade["close_timestamp"]
                        })
                        await self.close_trade(exchange, pair, trade, open_trades)
                    self.execution_processor.reset_active_capital()
                    self.daily_profit_loss += daily_profit
                    await self.nucleus.publicar_alerta({
                        "tipo": "cierre_diario",
                        "plugin_id": "crypto_trading",
                        "profit_loss": daily_profit,
                        "total_profit_loss": self.daily_profit_loss,
                        "capital": self.capital,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    await self.trading_db.save_report(
                        date=now.strftime("%Y-%m-%d"),
                        total_profit=daily_profit,
                        roi_percent=(daily_profit / self.capital) * 100 if self.capital != 0 else 0,
                        total_trades=len(open_trades),
                        report_data={"open_trades": 0},
                        timestamp=now.timestamp()
                    )
                    self.logger.info(f"[SettlementProcessor] Cierre diario: Ganancia/Pérdida: ${daily_profit}, Total: ${self.daily_profit_loss}, Capital: ${self.capital}")
                    await asyncio.sleep(86400 - 60)
                else:
                    await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"[SettlementProcessor] Error en cierre diario: {e}")
                await asyncio.sleep(60)
