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
        macro_cfg = self.config.get("macro_config", {})
        self.symbols = macro_cfg.get("symbols", [])
        self.altcoin_symbols = macro_cfg.get("altcoin_symbols", [])
        self.update_interval = macro_cfg.get("update_interval", 300)
        self.api_keys = macro_cfg.get("api_keys", {})
        breaker_cfg = macro_cfg.get("circuit_breaker", {})
        self.circuit_breakers = {
            symbol: CircuitBreaker(
                breaker_cfg.get("max_failures", 3),
                breaker_cfg.get("reset_timeout", 900)
            )
            for symbol in self.symbols + self.altcoin_symbols + ["DXY", "news"]
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
                            f"Alpha Vantage error para {symbol}: {response.status}"
                        )
                        self.circuit_breakers[symbol].register_failure()
                        return {}
                    result = await response.json()
                    quote = result.get("Global Quote", {})
                    price = float(quote.get("05. price", 0))
                    change = quote.get("10. change percent", "0").replace("%", "")
                    return {
                        "price": price,
                        "change_percent": float(change)
                    }
        except Exception as e:
            self.logger.error(f"Error en {symbol}: {e}")
            self.circuit_breakers[symbol].register_failure()
            return {}

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def fetch_dxy(self) -> Dict[str, Any]:
        if not self.circuit_breakers["DXY"].check():
            return {}
        try:
            async with aiohttp.ClientSession() as session:
                url = (
                    "https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE"
                    "&from_currency=USD&to_currency=DX-Y.NYB"
                    f"&apikey={self.api_keys['alpha_vantage']}"
                )
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(
                            f"Alpha Vantage error para DXY: {response.status}"
                        )
                        self.circuit_breakers["DXY"].register_failure()
                        return {}
                    result = await response.json()
                    rate = result["Realtime Currency Exchange Rate"]
                    price = float(rate["5. Exchange Rate"])
                    prev = self.macro_data_cache.get("DXY", {}).get("price", price)
                    change = ((price - prev) / prev * 100) if prev else 0
                    return {"price": price, "change_percent": change}
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
                symbols = ",".join(self.altcoin_symbols)
                url = (
                    "https://pro-api.coinmarketcap.com/v1/"
                    f"cryptocurrency/quotes/latest?symbol={symbols}"
                )
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        total = sum(
                            result["data"][s]["quote"]["USD"]["volume_24h"]
                            for s in result["data"]
                        )
                        return {
                            "altcoins": list(result["data"].keys()),
                            "altcoins_volume": total
                        }
                    self.logger.error(
                        f"CoinMarketCap error: {response.status}"
                    )
                    self.circuit_breakers["altcoins"].register_failure()
                    return {}
        except Exception as e:
            self.logger.error(f"Altcoins error: {e}")
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
                    self.logger.error(f"NewsAPI error: {response.status}")
                    self.circuit_breakers["news"].register_failure()
                    return {}
        except Exception as e:
            self.logger.error(f"News sentiment error: {e}")
            self.circuit_breakers["news"].register_failure()
            return {}

    async def sync_macro_data(self):
        while True:
            now = datetime.utcnow()
            if 7 <= now.hour <= 17:
                macro = {}
                for sym in self.symbols:
                    data = await self.fetch_alpha_vantage(sym)
                    if data:
                        key = sym.lower().replace("^", "")
                        macro[f"{key}_price"] = data["price"]
                        macro[f"{key}_change_percent"] = data["change_percent"]
                dxy = await self.fetch_dxy()
                if dxy:
                    macro["dxy_price"] = dxy["price"]
                    macro["dxy_change_percent"] = dxy["change_percent"]
                alt = await self.fetch_altcoins()
                if alt:
                    macro.update(alt)
                news = await self.fetch_news_sentiment()
                if news:
                    macro.update(news)
                macro["timestamp"] = datetime.utcnow().timestamp()
                self.macro_data_cache = macro
                if (
                    "dxy_change_percent" in macro and
                    "sp500_change_percent" in macro
                ):
                    dxy_c = macro["dxy_change_percent"]
                    sp500_c = macro["sp500_change_percent"]
                    macro["dxy_sp500_correlation"] = -0.5 if dxy_c * sp500_c < 0 else 0.5
                    macro["dxy_btc_correlation"] = -0.6 if dxy_c > 0 else 0.4
                await self.plugin_db.save_macro_data(macro)
                if "dxy_change_percent" in macro:
                    dxy_c = macro["dxy_change_percent"]
                    if abs(dxy_c) > 1:
                        await self.nucleus.publicar_alerta({
                            "tipo": "dxy_change",
                            "plugin": "crypto_trading",
                            "message": (
                                f"DXY cambió {dxy_c:.2f}%, "
                                f"riesgo {'alto' if dxy_c > 0 else 'bajo'}"
                            )
                        })
                self.logger.debug(f"Macro sync: {macro}")
            await asyncio.sleep(self.update_interval)

    async def manejar_evento(self, mensaje: Dict[str, Any]):
        try:
            if mensaje.get("tipo") == "macro_data":
                self.macro_data_cache[mensaje["tipo"]] = mensaje
                self.logger.debug(f"Macro data recibido: {mensaje}")
        except Exception as e:
            self.logger.error(f"Error evento macro: {e}")
            self.circuit_breakers["event"].register_failure()

    async def detener(self):
        await self.plugin_db.disconnect()
        self.logger.info("MacroProcessor detenido")
