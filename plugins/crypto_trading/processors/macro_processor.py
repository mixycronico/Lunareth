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
        self.critical_news = False  # Bandera para eventos críticos

    async def fetch_with_retries(self, fetch_func, *args, max_retries=3):
        for attempt in range(max_retries):
            try:
                return await fetch_func(*args)
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"Fallo al obtener datos después de {max_retries} intentos: {e}")
                    raise
                wait_time = 2 ** attempt
                self.logger.warning(f"Fallo en intento {attempt+1}, reintentando en {wait_time} segundos: {e}")
                await asyncio.sleep(wait_time)
        return None

    async def fetch_critical_news(self):
        """Simula la obtención de noticias críticas (en producción, usar una API real)."""
        try:
            # Simulación de noticias críticas (por ejemplo, decisiones de la FED)
            news_data = {"fed_rate_change": random.choice([0, 0.25, -0.25]), "timestamp": datetime.utcnow().isoformat()}
            self.critical_news = news_data["fed_rate_change"] != 0
            await self.redis.set("critical_news", json.dumps(news_data))
            self.logger.info(f"Noticias críticas obtenidas: {news_data}")
            return news_data
        except Exception as e:
            self.logger.error(f"Error al obtener noticias críticas: {e}")
            return None

    async def fetch_and_publish_data(self):
        if not self.cb.check():
            self.logger.warning("Circuit breaker activo, omitiendo obtención de datos macroeconómicos")
            return {"status": "skipped", "motivo": "circuito_abierto"}

        try:
            macro_data = await self.fetch_with_retries(self.alpha_vantage.fetch_macro_data)
            if not macro_data:
                macro_data = {"sp500": 0.0, "nasdaq": 0.0, "dxy": 0.0, "gold": 0.0, "oil": 0.0}

            crypto_data = {}
            for symbol in self.config.get("symbols", ["BTC/USDT", "ETH/USDT"]):
                data = await self.fetch_with_retries(self.coinmarketcap.fetch_crypto_data, symbol)
                crypto_data[symbol] = data if data else {"volume": 0, "market_cap": 0}

            # Ajustar datos según noticias críticas
            news_data = await self.redis.get("critical_news")
            if news_data:
                news_data = json.loads(news_data)
                if news_data["fed_rate_change"] > 0:  # Subida de tasas
                    macro_data["dxy"] += 0.5  # Aumentar DXY para reflejar fortaleza del dólar
                elif news_data["fed_rate_change"] < 0:  # Bajada de tasas
                    macro_data["dxy"] -= 0.5  # Reducir DXY

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
        """Bucle para obtener datos macroeconómicos, criptomonedas y noticias cada 60 segundos."""
        while True:
            try:
                await self.fetch_and_publish_data()
                await self.fetch_critical_news()
            except Exception as e:
                self.logger.error(f"Error en el bucle de obtención de datos: {e}")
            await asyncio.sleep(60)
