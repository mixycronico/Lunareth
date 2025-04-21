import logging
import json
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker
from plugins.crypto_trading.utils.settlement_utils import calcular_ganancias, registrar_historial
import numpy as np

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
        self.historical_profits = []
        self.trades_history = []
        self.is_paused = False
        self.recovery_capital_percentage = 0.1
        self.open_trades = {}  # Ahora gestionado directamente por SettlementProcessor

    async def update_capital_after_trade(self, side: str, trade_result: Dict[str, Any]):
        if side == "buy":
            self.capital -= trade_result["cantidad"]
        elif side == "sell":
            self.capital += trade_result["cantidad"]
        self.strategy.update_capital(self.capital)
        self.execution_processor.update_capital(self.capital)
        self.capital_processor.actualizar_total_capital(self.capital)
        self.logger.info(f"[SettlementProcessor] Capital actualizado: ${self.capital}")

    async def close_trade(self, exchange: str, pair: str, trade: dict):
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
        self.historical_profits.append(profit)
        self.trades_history.append({
            "profit": profit,
            "timestamp": trade["close_timestamp"],
            "is_win": profit > 0
        })
        self.strategy.update_capital(self.capital)
        self.execution_processor.update_capital(self.capital)
        self.capital_processor.actualizar_total_capital(self.capital)
        del self.open_trades[trade_id]
        await self.nucleus.publicar_alerta({
            "tipo": "operacion_cerrada",
            "plugin_id": "crypto_trading",
            "exchange": exchange,
            "pair": pair,
            "timestamp": trade["close_timestamp"]
        })

    async def micro_cycle_loop(self):
        while True:
            try:
                now = datetime.datetime.now()
                if 6 <= now.hour < 22 and now.hour % 6 == 0 and now.minute == 0:
                    micro_profit = 0
                    for trade_id, trade in list(self.open_trades.items()):
                        exchange, pair = trade_id.split(":")
                        current_price = 50000
                        profit = trade["cantidad"] * (current_price - trade["precio"])
                        micro_profit += profit
                        await registrar_historial(self.redis, f"trade_history:{exchange}:{pair}", {
                            "precio_entrada": trade["precio"],
                            "precio_salida": current_price,
                            "cantidad": trade["cantidad"],
                            "timestamp": trade["close_timestamp"]
                        })
                        await self.close_trade(exchange, pair, trade)
                    self.logger.info(f"[SettlementProcessor] Micro-ciclo a las {now.hour}:00: Ganancia/Pérdida: ${micro_profit}")
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"[SettlementProcessor] Error en micro-ciclo: {e}")
                await asyncio.sleep(60)

    async def daily_close_loop(self):
        while True:
            try:
                now = datetime.datetime.now()
                close_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
                if now.hour == 22 and now.minute == 0:
                    daily_profit = 0
                    for trade_id, trade in list(self.open_trades.items()):
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
                        await self.close_trade(exchange, pair, trade)
                    self.execution_processor.reset_active_capital()
                    self.daily_profit_loss += daily_profit
                    await self.capital_processor.update_strategy_performance(daily_profit)

                    sharpe_ratio = self.calculate_sharpe_ratio()
                    max_drawdown = self.calculate_max_drawdown()
                    win_rate = self.calculate_win_rate()
                    trades_per_day = self.calculate_trades_per_day()

                    report = {
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                        "daily_profit": daily_profit,
                        "total_profit_loss": self.daily_profit_loss,
                        "capital": self.capital,
                        "win_rate": win_rate,
                        "trades_per_day": trades_per_day
                    }
                    await self.redis.set(f"historical_report:{now.strftime('%Y-%m-%d')}", json.dumps(report))

                    await self.nucleus.publicar_alerta({
                        "tipo": "cierre_diario",
                        "plugin_id": "crypto_trading",
                        "profit_loss": daily_profit,
                        "total_profit_loss": self.daily_profit_loss,
                        "capital": self.capital,
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                        "win_rate": win_rate,
                        "trades_per_day": trades_per_day,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    await self.trading_db.save_report(
                        date=now.strftime("%Y-%m-%d"),
                        total_profit=daily_profit,
                        roi_percent=(daily_profit / self.capital) * 100 if self.capital != 0 else 0,
                        total_trades=len(self.open_trades),
                        report_data={"open_trades": 0, "sharpe_ratio": sharpe_ratio, "max_drawdown": max_drawdown, "win_rate": win_rate, "trades_per_day": trades_per_day},
                        timestamp=now.timestamp()
                    )
                    self.logger.info(f"[SettlementProcessor] Cierre diario: Ganancia/Pérdida: ${daily_profit}, Total: ${self.daily_profit_loss}, Capital: ${self.capital}")

                    await self.cleanup_old_data(now)

                    await asyncio.sleep(86400 - 60)
                else:
                    await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"[SettlementProcessor] Error en cierre diario: {e}")
                await asyncio.sleep(60)

    async def cleanup_old_data(self, now):
        try:
            keys = await self.redis.keys("historical_report:*")
            for key in keys:
                date_str = key.split(":")[-1]
                report_date = datetime.strptime(date_str, "%Y-%m-%d")
                if (now - report_date).days > 1:
                    await self.redis.delete(key)
                    self.logger.info(f"Datos antiguos eliminados: {key}")
        except Exception as e:
            self.logger.error(f"Error al limpiar datos antiguos en Redis: {e}")

    def calculate_sharpe_ratio(self):
        if len(self.historical_profits) < 2:
            return 0.0
        returns = np.array(self.historical_profits) / self.capital
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        sharpe_ratio = mean_return / std_return * np.sqrt(252)
        return sharpe_ratio

    def calculate_max_drawdown(self):
        if len(self.historical_profits) < 2:
            return 0.0
        cumulative = np.cumsum(self.historical_profits)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        return max_drawdown

    def calculate_win_rate(self):
        if not self.trades_history:
            return 0.0
        wins = sum(1 for trade in self.trades_history if trade["is_win"])
        return (wins / len(self.trades_history)) * 100

    def calculate_trades_per_day(self):
        if not self.trades_history:
            return 0.0
        if len(self.trades_history) < 2:
            return len(self.trades_history)
        timestamps = [datetime.fromisoformat(trade["timestamp"]) for trade in self.trades_history]
        days = (timestamps[-1] - timestamps[0]).days + 1
        return len(self.trades_history) / days

    async def handle_market_crash(self):
        while True:
            try:
                market_data = await self.redis.get("market_data")
                if not market_data:
                    await asyncio.sleep(60)
                    continue
                market_data = json.loads(market_data)
                macro_data = market_data["macro"]
                crypto_data = market_data["crypto"]

                btc_change = next((data["price_change"] for data in crypto_data.values() if "BTC" in data["symbol"]), 0)
                sp500_change = macro_data.get("sp500", 0.0)
                if btc_change < -0.2 or sp500_change < -0.2:
                    self.logger.warning("Caída del mercado detectada: pausando operaciones por 1 hora")
                    self.is_paused = True
                    self.recovery_capital_percentage = 0.1

                    alert = {
                        "tipo": "evento_critico",
                        "plugin_id": "crypto_trading",
                        "evento": "caida_mercado",
                        "detalle": f"Caída detectada: BTC {btc_change*100:.2f}%, SP500 {sp500_change*100:.2f}%",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    await self.redis.xadd("critical_events", {"data": json.dumps(alert)})
                    await self.nucleus.publicar_alerta(alert)

                    for trade_id, trade in list(self.open_trades.items()):
                        exchange, pair = trade_id.split(":")
                        await self.close_trade(exchange, pair, trade)

                    for i in range(6):
                        await asyncio.sleep(600)
                        self.recovery_capital_percentage = min(1.0, self.recovery_capital_percentage + 0.15)
                        self.logger.info(f"Recuperación gradual: {self.recovery_capital_percentage*100:.2f}% del capital disponible")
                    self.is_paused = False
                else:
                    await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"Error al manejar caída del mercado: {e}")
                await asyncio.sleep(60)
