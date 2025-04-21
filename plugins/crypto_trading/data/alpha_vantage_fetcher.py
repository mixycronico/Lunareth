import aiohttp
import logging
from typing import Dict

class AlphaVantageFetcher:
    def __init__(self, api_key: str):
        self.logger = logging.getLogger("AlphaVantageFetcher")
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    async def fetch_macro_data(self) -> Dict[str, float]:
        """Obtiene datos macroeconómicos de AlphaVantage."""
        try:
            async with aiohttp.ClientSession() as session:
                # Lista de símbolos macroeconómicos a obtener
                symbols = {
                    "sp500": "INDEX:SPX",    # SP500
                    "nasdaq": "INDEX:NASX",  # Nasdaq
                    "dxy": "CURRENCY:USD",   # DXY
                    "gold": "COMMODITY:GOLD", # Oro
                    "oil": "COMMODITY:OIL"   # Petróleo
                }
                macro_data = {}

                for key, symbol in symbols.items():
                    params = {
                        "function": "TIME_SERIES_DAILY",
                        "symbol": symbol,
                        "apikey": self.api_key,
                        "outputsize": "compact"
                    }
                    async with session.get(self.base_url, params=params) as response:
                        if response.status != 200:
                            self.logger.warning(f"Fallo al obtener datos de {symbol}: {response.status}")
                            macro_data[key] = 0.0
                            continue

                        data = await response.json()
                        if "Time Series (Daily)" not in data:
                            self.logger.warning(f"Datos no disponibles para {symbol}")
                            macro_data[key] = 0.0
                            continue

                        # Obtener el cambio porcentual del último día
                        time_series = data["Time Series (Daily)"]
                        dates = sorted(time_series.keys())
                        latest = time_series[dates[0]]
                        previous = time_series[dates[1]]
                        latest_close = float(latest["4. close"])
                        previous_close = float(previous["4. close"])
                        change_percent = ((latest_close - previous_close) / previous_close) * 100
                        macro_data[key] = change_percent

                self.logger.info(f"Datos macroeconómicos obtenidos: {macro_data}")
                return macro_data
        except Exception as e:
            self.logger.error(f"Error al obtener datos macroeconómicos: {e}")
            return {key: 0.0 for key in ["sp500", "nasdaq", "dxy", "gold", "oil"]}
