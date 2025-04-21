import aiohttp
import logging
from typing import Dict

class CoinMarketCapFetcher:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger("CoinMarketCapFetcher")
        self.api_key = config.get("macro_config", {}).get("api_keys", {}).get("coinmarketcap", "default_key")
        self.base_url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency"

    async def fetch_crypto_data(self, symbol: str) -> Dict[str, float]:
        """Obtiene datos de criptomonedas de CoinMarketCap."""
        try:
            async with aiohttp.ClientSession() as session:
                crypto_symbol = symbol.split("/")[0]
                headers = {
                    "Accept": "application/json",
                    "X-CMC_PRO_API_KEY": self.api_key
                }
                params = {
                    "symbol": crypto_symbol,
                    "convert": "USD"
                }
                async with session.get(f"{self.base_url}/quotes/latest", headers=headers, params=params) as response:
                    if response.status != 200:
                        self.logger.warning(f"Fallo al obtener datos de {symbol}: {response.status}")
                        return {"volume": 0.0, "market_cap": 0.0}

                    data = await response.json()
                    if "data" not in data or crypto_symbol not in data["data"]:
                        self.logger.warning(f"Datos no disponibles para {symbol}")
                        return {"volume": 0.0, "market_cap": 0.0}

                    crypto_data = data["data"][crypto_symbol]["quote"]["USD"]
                    result = {
                        "volume": float(crypto_data["volume_24h"]),
                        "market_cap": float(crypto_data["market_cap"])
                    }
                    self.logger.info(f"Datos de criptomoneda obtenidos para {symbol}: {result}")
                    return result
        except Exception as e:
            self.logger.error(f"Error al obtener datos de criptomoneda para {symbol}: {e}")
            return {"volume": 0.0, "market_cap": 0.0}
