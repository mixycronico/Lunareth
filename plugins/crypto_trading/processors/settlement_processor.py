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
        self.trade_history = []  # Para reportes históricos
        self.micro_cycle_interval = 6 * 3600  # 6 horas en segundos

    async def update_capital_after_trade(self, side: str, trade_result: Dict[str, Any]):
        if side == "buy":
            self.capital -= trade_result["cantidad"]
        elif side == "sell":
            self.capital += trade_result["cantidad"]
        self.strategy.update_capital(self.capital)
        self.execution_processor.update_capital(self.capital)
        self.capital_processor.actualizar_total_capital(self.capital)

        # Agregar operación al historial
        self.trade_history.append({
            "side": side,
            "price": trade_result["precio"],
            "quantity": trade_result["cantidad"],
            "timestamp": trade_result["timestamp"]
        })
        if len(self.trade_history) > 1000:
            self.trade_history.pop(0)
        self.logger.info(f"[SettlementProcessor] Capital actualizado: ${self.capital}")

    async def close_trade(self, exchange: str, pair: str, trade: dict, open_trades: Dict[str, Dict]):
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

    async def detect_market_crash(self):
        """Detecta caídas extremas del mercado (mayor al 20% en un día)."""
        market_data = await self.redis.get("market_data")
        if not market_data:
            return False

        market_data = json.loads(market_data)
        crypto_data = market_data["crypto"]
        avg_price_change = sum(
            (data["price_change"] if "price_change" in data else 0) for data in crypto_data.values()
        ) / len(crypto_data)

        if avg_price_change < -0.20:  # Caída mayor al 20%
            self.logger.warning("Caída extrema detectada, pausando operaciones por 1 hora")
            await asyncio.sleep(3600)  # Pausar por 1 hora
            return True
        return False

    async def micro_cycle(self, open_trades: Dict[str, Dict]):
        """Realiza un cierre parcial cada 6 horas dentro del horario de trading."""
        while True:
            try:
                now = datetime.datetime.now()
                start_time = now.replace(hour=6, minute=0, second=0, microsecond=0)
                end_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
                within_trading_hours = start_time <= now <= end_time

                if within_trading_hours and now.hour % 6 == 0 and now.minute == 0:
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
                    self.daily_profit_loss += daily_profit
                    await self.nucleus.publicar_alerta({
                        "tipo": "micro_ciclo",
                        "plugin_id": "crypto_trading",
                        "profit_loss": daily_profit,
                        "total_profit_loss": self.daily_profit_loss,
                        "capital": self.capital,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    self.logger.info(f"[SettlementProcessor] Micro-ciclo: Ganancia/Pérdida: ${daily_profit}, Total: ${self.daily_profit_loss}, Capital: ${self.capital}")
                await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"[SettlementProcessor] Error en micro-ciclo: {e}")
                await asyncio.sleep(60)

    async def daily_close_loop(self, open_trades: Dict[str, Dict]):
        while True:
            try:
                if await self.detect_market_crash():
                    continue

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

                    # Generar reporte histórico
                    sharpe_ratio, max_drawdown = self.calculate_historical_metrics()
                    report = {
                        "sharpe_ratio": sharpe_ratio,
                        "max_drawdown": max_drawdown,
                        "total_trades": len(self.trade_history),
                        "win_rate": sum(1 for trade in self.trade_history if trade["profit"] > 0) / len(self.trade_history) if self.trade_history else 0
                    }
                    await self.redis.set(f"historical_report:{now.strftime('%Y-%m-%d')}", json.dumps(report))

                    await self.nucleus.publicar_alerta({
                        "tipo": "cierre_diario",
                        "plugin_id": "crypto_trading",
                        "profit_loss": daily_profit,
                        "total_profit_loss": self.daily_profit_loss,
                        "capital": self.capital,
                        "report": report,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    await self.trading_db.save_report(
                        date=now.strftime("%Y-%m-%d"),
                        total_profit=daily_profit,
                        roi_percent=(daily_profit / self.capital) * 100 if self.capital != 0 else 0,
                        total_trades=len(open_trades),
                        report_data={"open_trades": 0, "historical_metrics": report},
                        timestamp=now.timestamp()
                    )
                    self.logger.info(f"[SettlementProcessor] Cierre diario: Ganancia/Pérdida: ${daily_profit}, Total: ${self.daily_profit_loss}, Capital: ${self.capital}")
                    await asyncio.sleep(86400 - 60)
                else:
                    await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"[SettlementProcessor] Error en cierre diario: {e}")
                await asyncio.sleep(60)

    def calculate_historical_metrics(self):
        """Calcula métricas históricas como Sharpe Ratio y Max Drawdown."""
        if not self.trade_history:
            return 0.0, 0.0

        profits = [trade["profit"] for trade in self.trade_history]
        returns = np.array(profits) / self.capital
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0

        cumulative_returns = np.cumsum(profits)
        max_drawdown = 0
        peak = cumulative_returns[0]
        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            drawdown = (peak - ret) / peak if peak != 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return sharpe_ratio, max_drawdown
