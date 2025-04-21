import logging
import aiohttp
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker

class MonitorProcessor:
    def __init__(self, config, redis, open_trades=None):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("MonitorProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 600)
        )
        self.threshold = config.get("volatility_threshold", 0.025)
        self.symbols = config.get("symbols", ["BTC/USDT", "ETH/USDT"])
        self.open_trades = open_trades  # Referencia a self.open_trades de main.py

    async def analizar_volatilidad(self):
        if not self.cb.check():
            self.logger.warning("Circuit breaker activo, omitiendo análisis de volatilidad")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            resultados = []
            async with aiohttp.ClientSession() as session:
                for symbol in self.symbols:
                    url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol.replace('/', '')}"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            price_change = abs(float(data["priceChangePercent"])) / 100
                            is_volatile = price_change >= self.threshold
                            resultados.append({
                                "symbol": symbol,
                                "volatilidad": price_change,
                                "alerta": is_volatile,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                        else:
                            self.logger.warning(f"Error al consultar {symbol}: {resp.status}")

            await self.redis.set("volatility_data", json.dumps(resultados))
            self.logger.info(f"Volatilidad analizada en {len(resultados)} símbolos")
            return {"status": "ok", "datos": resultados}
        except Exception as e:
            self.logger.error("Error en análisis de volatilidad", exc_info=True)
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}

    async def continuous_open_trades_monitor(self, close_trade_callback):
        """Monitorea continuamente las operaciones abiertas cada 30 segundos."""
        while True:
            try:
                for trade_id, trade in list(self.open_trades.items()):
                    exchange, pair = trade_id.split(":")
                    self.logger.info(f"[MonitorProcessor] Monitoreando operación abierta para {exchange}:{pair}: {trade}")
                    current_price = 50000  # Simulado, en producción obtener del exchange
                    if (trade["tipo"] == "buy" and current_price >= trade["take_profit"]) or \
                       (trade["tipo"] == "buy" and current_price <= trade["stop_loss"]) or \
                       (trade["tipo"] == "sell" and current_price <= trade["take_profit"]) or \
                       (trade["tipo"] == "sell" and current_price >= trade["stop_loss"]):
                        await close_trade_callback(exchange, pair, trade)
                await asyncio.sleep(30)
            except Exception as e:
                self.logger.error(f"[MonitorProcessor] Error en monitoreo continuo de operaciones abiertas: {e}")
                await asyncio.sleep(30)
