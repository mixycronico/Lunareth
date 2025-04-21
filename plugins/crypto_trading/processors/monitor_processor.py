import logging
import aiohttp
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker

class MonitorProcessor:
    def __init__(self, config, redis):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("MonitorProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 600)
        )
        self.threshold = config.get("volatility_threshold", 0.025)
        self.symbols = config.get("symbols", ["BTC/USDT", "ETH/USDT"])

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

            # Almacenar en Redis para que MonitorBlock lo use
            await self.redis.set("volatility_data", json.dumps(resultados))
            self.logger.info(f"Volatilidad analizada en {len(resultados)} símbolos")
            return {"status": "ok", "datos": resultados}
        except Exception as e:
            self.logger.error("Error en análisis de volatilidad", exc_info=True)
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}
