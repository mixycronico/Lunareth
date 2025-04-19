#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/processors/macro_processor.py
Sincroniza datos macroeconómicos (S&P 500, Nasdaq, VIX, oro, petróleo, altcoins)
desde APIs externas como AlphaVantage, CoinMarketCap y NewsAPI.
"""

from corec.core import ComponenteBase
from ..utils.db import TradingDB
from ..utils.helpers import CircuitBreaker
import aiohttp
import asyncio
import logging
import backoff
from typing import Dict, Any
from datetime import datetime


class MacroProcessor(ComponenteBase):
    def __init__(self, config: Dict[str, Any], redis_client):
        super().__init__()
        self.config = config.get("crypto_trading", {})
        self.redis_client = redis_client
        self.logger = logging.getLogger("MacroProcessor")
        self.symbols = self.config.get("macro_config", {}).get("symbols", [])
        self.altcoin_symbols = self.config.get("macro_config", {}).get("altcoin_symbols", [])
        self.update_interval = self.config.get("macro_config", {}).get("update_interval", 300)
        self.api_keys = self.config.get("macro_config", {}).get("api_keys", {})
        self.circuit_breakers = {
            symbol: CircuitBreaker(
                self.config.get("macro_config", {}).get("circuit_breaker", {}).get("max_failures", 3),
                self.config.get("macro_config", {}).get("circuit_breaker", {}).get("reset_timeout", 900)
            ) for symbol in self.symbols + self.altcoin_symbols + ["DXY", "news"]
        }
        self.plugin_db = TradingDB(self.config.get("db_config", {}))
        self.macro_data_cache = {}

    async def inicializar(self):
        await self.plugin_db.connect()
        asyncio.create_task(self.sync_macro_data())
        self.logger.info("MacroProcessor inicializado")

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def fetch_alpha_vantage(self, symbol: str) -> Dict[str, Any]:
        if not self.circuit_breakers[symbol].check():
            return {}
        try:
            async with aiohttp.ClientSession() as session:
                url = (
                    "https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
                    f"&symbol={symbol}&apikey={self.api_keys['alpha_vantage']}"
                )
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"Error en Alpha Vantage para {symbol}: {response.status}"
                        )
                        self.circuit_breakers[symbol].register_failure()
                        return {}
                    result = await response.json()
                    price = float(result["Global Quote"]["05. price"])
                    change_percent = float(
                        result["Global Quote"]["10. change percent"].replace("%", "")
                    )
                    return {"price": price, "change_percent": change_percent}
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de {symbol}: {e}")
            self.circuit_breakers[symbol].register_failure()
            return {}

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def fetch_dxy(self) -> Dict[str, Any]:
        if not self.circuit_breakers["DXY"].check():
            return {}
        try:
            async with aiohttp.ClientSession() as session:
                url = (
                    "https://www.alphavantage.co/query?"
                    "function=CURRENCY_EXCHANGE_RATE"
                    "&from_currency=USD&to_currency=DX-Y.NYB"
                    f"&apikey={self.api_keys['alpha_vantage']}"
                )
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"Error en Alpha Vantage para DXY: {response.status}"
                        )
                        self.circuit_breakers["DXY"].register_failure()
                        return {}
                    result = await response.json()
                    price = float(result["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
                    previous_price = self.macro_data_cache.get("DXY", {}).get("price", price)
                    change_percent = (
                        (price - previous_price) / previous_price * 100
                        if previous_price
                        else 0
                    )
                    return {"price": price, "change_percent": change_percent}
        except Exception as e:
            self.logger.error(f"Error obteniendo DXY: {e}")
            self.circuit_breakers["DXY"].register_failure()
            return {}

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def fetch_altcoins(self) -> Dict[str, Any]:
        if not self.circuit_breakers["altcoins"].check():
            return {}
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"X-CMC_PRO_API_KEY": self.api_keys["coinmarketcap"]}
                url = (
                    "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest?"
                    f"symbol={','.join(self.altcoin_symbols)}"
                )
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "altcoins": list(result["data"].keys()),
                            "altcoins_volume": sum(
                                result["data"][s]["quote"]["USD"]["volume_24h"]
                                for s in result["data"]
                            ),
                        }
                    else:
                        self.logger.error(f"Error en CoinMarketCap: {response.status}")
                        self.circuit_breakers["altcoins"].register_failure()
                        return {}
        except Exception as e:
            self.logger.error(f"Error obteniendo altcoins: {e}")
            self.circuit_breakers["altcoins"].register_failure()
            return {}

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def fetch_news_sentiment(self) -> Dict[str, Any]:
        if not self.circuit_breakers["news"].check():
            return {}
        try:
            async with aiohttp.ClientSession() as session:
                url = (
                    "https://newsapi.org/v2/everything?"
                    f"q=cryptocurrency&apiKey={self.api_keys['newsapi']}"
                )
                async with session.get(url) as response:
                    if response.status == 200:
                        await response.json()
                        return {"news_sentiment": 0.7}
                    else:
                        self.logger.error(f"Error en NewsAPI: {response.status}")
                        self.circuit_breakers["news"].register_failure()
                        return {}
        except Exception as e:
            self.logger.error(f"Error obteniendo sentimiento de noticias: {e}")
            self.circuit_breakers["news"].register_failure()
            return {}

    async def sync_macro_data(self):
        while True:
            now = datetime.utcnow()
            if 7 <= now.hour <= 17:
                macro_data = {}
                for symbol in self.symbols:
                    data = await self.fetch_alpha_vantage(symbol)
                    if data:
                        key = symbol.lower().replace("^", "")
                        macro_data[f"{key}_price"] = data["price"]
                        macro_data[f"{key}_change_percent"] = data["change_percent"]

                dxy_data = await self.fetch_dxy()
                if dxy_data:
                    macro_data["dxy_price"] = dxy_data["price"]
                    macro_data["dxy_change_percent"] = dxy_data["change_percent"]

                altcoin_data = await self.fetch_altcoins()
                if altcoin_data:
                    macro_data.update(altcoin_data)

                news_data = await self.fetch_news_sentiment()
                if news_data:
                    macro_data.update(news_data)

                macro_data["timestamp"] = datetime.utcnow().timestamp()
                self.macro_data_cache = macro_data

                if (
                    "dxy_change_percent" in macro_data
                    and "sp500_change_percent" in macro_data
                ):
                    macro_data["dxy_sp500_correlation"] = (
                        -0.5
                        if macro_data["dxy_change_percent"]
                        * macro_data["sp500_change_percent"] < 0
                        else 0.5
                    )
                    macro_data["dxy_btc_correlation"] = (
                        -0.6 if macro_data["dxy_change_percent"] > 0 else 0.4
                    )

                await self.redis_client.xadd(
                    "crypto_trading_data",
                    {
                        "data": (
                            f"{macro_data['timestamp']}|macro_data|0.0|1|"
                            + str(macro_data)
                        )
                    },
                )
                await self.plugin_db.save_macro_data(macro_data)

                if (
                    "dxy_change_percent" in macro_data
                    and abs(macro_data["dxy_change_percent"]) > 1
                ):
                    await self.nucleus.publicar_alerta({
                        "tipo": "dxy_change",
                        "plugin": "crypto_trading",
                        "message": (
                            f"DXY cambió {macro_data['dxy_change_percent']:.2f}%, "
                            f"riesgo {'alto' if macro_data['dxy_change_percent'] > 0 else 'bajo'}"
                        )
                    })

                self.logger.debug(f"Datos macro sincronizados: {macro_data}")

            await asyncio.sleep(self.update_interval)

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            if mensaje.get("tipo") == "macro_data":
                self.macro_data_cache[mensaje["tipo"]] = mensaje
                self.logger.debug(f"Datos macro recibidos: {mensaje}")
        except Exception as e:
            self.logger.error(f"Error procesando evento: {e}")
            self.circuit_breakers["event"].register_failure()

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("MacroProcessor detenido")
