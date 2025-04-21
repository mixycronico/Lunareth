import logging
import aiohttp
from datetime import datetime
from plugins.crypto_trading.utils.helpers import CircuitBreaker
from plugins.crypto_trading.data.alpha_vantage_fetcher import AlphaVantageFetcher
from plugins.crypto_trading.data.coinmarketcap_fetcher import CoinMarketCapFetcher
import json
import asyncio

class MacroProcessor:
    def __init__(self, config, redis):
        self.config = config
        self.redis = redis
        self.logger = logging.getLogger("MacroProcessor")
        self.cb = CircuitBreaker(
            max_failures=config.get("cb_max_failures", 3),
            reset_timeout=config.get("cb_reset_timeout", 900)
        )
        self.alpha_vantage = AlphaVantageFetcher(config)
        self.coinmarketcap = CoinMarketCapFetcher(config)

    async def fetch_with_retries(self, fetch_func, max_retries=5):
        """Realiza una solicitud con reintentos exponenciales."""
        for attempt in range(max_retries):
            try:
                return await fetch_func()
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Error tras {max_retries} intentos: {e}")
                    self.cb.register_failure()
                    return None
                delay = 2 ** attempt
                self.logger.warning(f"Intento {attempt + 1} fallido, reintentando en {delay} segundos: {e}")
                await asyncio.sleep(delay)
        return None

    async def fetch_and_publish_data(self):
        if not self.cb.check():
            self.logger.warning("Circuit breaker activo, omitiendo obtención de datos macroeconómicos")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            # Obtener datos macroeconómicos con reintentos
            macro_data = await self.fetch_with_retries(self.alpha_vantage.fetch_macro_data)
            if macro_data is None:
                return {"status": "error", "motivo": "failed_to_fetch_macro_data"}

            # Obtener datos de criptomonedas con reintentos
            crypto_data = {}
            for symbol in self.config.get("symbols", ["BTC/USDT", "ETH/USDT"]):
                data = await self.fetch_with_retries(lambda s=symbol: self.coinmarketcap.fetch_crypto_data(s))
                if data:
                    crypto_data[symbol] = data
                else:
                    crypto_data[symbol] = {"volume": 0, "market_cap": 0}

            combined_data = {
                "macro": macro_data,
                "crypto": crypto_data,
                "timestamp": datetime.utcnow().isoformat()
            }
            await self.redis.set("market_data", json.dumps(combined_data))
            self.logger.info("Datos macroeconómicos y de criptomonedas publicados en Redis")
            return {"status": "ok", "datos": combined_data}
        except Exception as e:
            self.logger.error("Error al obtener datos macroeconómicos", exc_info=True)
            self.cb.register_failure()
            return {"status": "error", "motivo": str(e)}

    async def data_fetch_loop(self):
        while True:
            try:
                await self.fetch_and_publish_data()
            except Exception as e:
                self.logger.error(f"Error en el bucle de obtención de datos: {e}")
            await asyncio.sleep(60)
