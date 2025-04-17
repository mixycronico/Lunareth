#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/macro_sync/processors/macro_processor.py
"""
macro_processor.py
Sincroniza datos macroeconómicos (S&P 500, Nasdaq, VIX, oro, petróleo, altcoins, DXY) desde APIs externas.
Publica en macro_data.
"""

from ....core.processors.base import ProcesadorBase
from ....core.entidad_base import Event
from ....utils.logging import logger
from ..utils.db import MacroDB
import aiohttp
import asyncio
import json
import zstandard as zstd
from typing import Dict, Any
from datetime import datetime, timedelta
import backoff

class MacroProcessor(ProcesadorBase):
    def __init__(self, config: Dict[str, Any], redis_client, db_config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger("MacroProcessor")
        self.symbols = config.get("macro_config", {}).get("symbols", [])
        self.altcoin_symbols = config.get("macro_config", {}).get("altcoin_symbols", [])
        self.update_interval = config.get("macro_config", {}).get("update_interval", 300)
        self.api_keys = config.get("macro_config", {}).get("api_keys", {})
        self.circuit_breaker = config.get("config", {}).get("circuit_breaker", {})
        self.plugin_db = None
        self.failure_count = {symbol: 0 for symbol in self.symbols + self.altcoin_symbols + ["DXY"]}
        self.breaker_tripped = {symbol: False for symbol in self.symbols + self.altcoin_symbols + ["DXY"]}
        self.breaker_reset_time = {symbol: None for symbol in self.symbols + self.altcoin_symbols + ["DXY"]}
        self.macro_data_cache = {}  # Caché local para datos macro

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.plugin_db = MacroDB(self.db_config)
        if not await self.plugin_db.connect():
            self.logger.warning("No se pudo conectar a macro_db")
            await self.nucleus.publicar_alerta({"tipo": "db_connection_error", "plugin": "macro_sync", "message": "No se pudo conectar a macro_db"})
        asyncio.create_task(self.sync_macro_data())
        self.logger.info("MacroProcessor inicializado")

    async def check_circuit_breaker(self, symbol: str) -> bool:
        if self.breaker_tripped[symbol]:
            now = datetime.utcnow()
            if now >= self.breaker_reset_time[symbol]:
                self.breaker_tripped[symbol] = False
                self.failure_count[symbol] = 0
                self.breaker_reset_time[symbol] = None
                self.logger.info(f"Circuit breaker reseteado para {symbol}")
            else:
                self.logger.warning(f"Circuit breaker activo para {symbol} hasta {self.breaker_reset_time[symbol]}")
                return False
        return True

    async def register_failure(self, symbol: str) -> None:
        self.failure_count[symbol] += 1
        if self.failure_count[symbol] >= self.circuit_breaker.get("max_failures", 3):
            self.breaker_tripped[symbol] = True
            self.breaker_reset_time[symbol] = datetime.utcnow() + timedelta(seconds=self.circuit_breaker.get("reset_timeout", 900))
            self.logger.error(f"Circuit breaker activado para {symbol} hasta {self.breaker_reset_time[symbol]}")
            await self.nucleus.publicar_alerta({"tipo": "circuit_breaker_tripped", "plugin": "macro_sync", "symbol": symbol})

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def fetch_alpha_vantage(self, symbol: str) -> Dict[str, Any]:
        if not await self.check_circuit_breaker(symbol):
            return {}
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.api_keys['alpha_vantage']}"
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"Error en Alpha Vantage para {symbol}: {response.status}")
                        await self.register_failure(symbol)
                        return {}
                    data = await response.json()
                    price = float(data["Global Quote"]["05. price"])
                    change_percent = float(data["Global Quote"]["10. change percent"].replace("%", ""))
                    return {"price": price, "change_percent": change_percent}
        except Exception as e:
            self.logger.error(f"Error obteniendo datos de {symbol}: {e}")
            await self.register_failure(symbol)
            return {}

    @backoff.on_exception(backoff.expo, Exception, max_tries=5)
    async def fetch_dxy(self) -> Dict[str, Any]:
        if not await self.check_circuit_breaker("DXY"):
            return {}
        try:
            async with aiohttp.ClientSession() as session:
                url = f"https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=USD&to_currency=DX-Y.NYB&apikey={self.api_keys['alpha_vantage']}"
                async with session.get(url) as response:
                    if response.status != 200:
                        self.logger.error(f"Error en Alpha Vantage para DXY: {response.status}")
                        await self.register_failure("DXY")
                        return {}
                    data = await response.json()
                    price = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
                    # Simular cambio porcentual (Alpha Vantage no proporciona cambio directo para DXY)
                    previous_price = self.macro_data_cache.get("DXY", {}).get("price", price)
                    change_percent = ((price - previous_price) / previous_price * 100) if previous_price else 0
                    return {"price": price, "change_percent": change_percent}
        except Exception as e:
            self.logger.error(f"Error obteniendo DXY: {e}")
            await self.register_failure("DXY")
            return {}

    async def sync_macro_data(self):
        while True:
            now = datetime.now()
            if now.hour >= 7 and now.hour <= 17:  # Horario de Nueva York
                macro_data = {}
                for symbol in self.symbols:
                    data = await self.fetch_alpha_vantage(symbol)
                    if data:
                        macro_data[f"{symbol.lower().replace('^', '')}_price"] = data["price"]
                        macro_data[f"{symbol.lower().replace('^', '')}_change_percent"] = data["change_percent"]

                # Obtener DXY
                dxy_data = await self.fetch_dxy()
                if dxy_data:
                    macro_data["dxy_price"] = dxy_data["price"]
                    macro_data["dxy_change_percent"] = dxy_data["change_percent"]

                # Obtener altcoins (sin cambios)
                async with aiohttp.ClientSession() as session:
                    headers = {"X-CMC_PRO_API_KEY": self.api_keys["coinmarketcap"]}
                    url = f"https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest?symbol={','.join(self.altcoin_symbols)}"
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            macro_data["altcoins"] = list(data["data"].keys())
                            macro_data["altcoins_volume"] = sum(data["data"][s]["quote"]["USD"]["volume_24h"] for s in data["data"])
                        else:
                            self.logger.error(f"Error en CoinMarketCap: {response.status}")
                            await self.register_failure("altcoins")

                # Obtener sentimiento (sin cambios)
                async with aiohttp.ClientSession() as session:
                    url = f"https://newsapi.org/v2/everything?q=cryptocurrency&apiKey={self.api_keys['newsapi']}"
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            macro_data["news_sentiment"] = 0.7  # Simulado
                        else:
                            self.logger.error(f"Error en NewsAPI: {response.status}")
                            await self.register_failure("news")

                macro_data["timestamp"] = datetime.utcnow().timestamp()
                self.macro_data_cache = macro_data

                # Correlación DXY con S&P 500 y BTC
                if "dxy_change_percent" in macro_data and "sp500_change_percent" in macro_data:
                    macro_data["dxy_sp500_correlation"] = -0.5 if macro_data["dxy_change_percent"] * macro_data["sp500_change_percent"] < 0 else 0.5
                    macro_data["dxy_btc_correlation"] = -0.6 if macro_data["dxy_change_percent"] > 0 else 0.4  # Simulado

                # Publicar datos
                datos_comprimidos = zstd.compress(json.dumps(macro_data).encode())
                await self.redis_client.xadd("macro_data", {"data": datos_comprimidos})

                # Almacenar en base de datos
                if self.plugin_db and self.plugin_db.conn:
                    await self.plugin_db.save_macro_data(macro_data)

                # Generar alertas para DXY
                if "dxy_change_percent" in macro_data and abs(macro_data["dxy_change_percent"]) > 1:
                    await self.nucleus.publicar_alerta({
                        "tipo": "dxy_change",
                        "plugin": "macro_sync",
                        "message": f"DXY cambió {macro_data['dxy_change_percent']:.2f}%, riesgo {'alto' if macro_data['dxy_change_percent'] > 0 else 'bajo'}"
                    })

                self.logger.debug(f"Datos macro sincronizados: {macro_data}")
            await asyncio.sleep(self.update_interval)

    async def manejar_evento(self, event: Event) -> None:
        try:
            datos = json.loads(zstd.decompress(event.datos["data"]))
            self.macro_data_cache[event.canal] = datos
            self.logger.debug(f"Datos recibidos: {event.canal}")
        except Exception as e:
            self.logger.error(f"Error manejando evento: {e}")
            await self.register_failure("event")
            await self.nucleus.publicar_alerta({"tipo": "event_error", "plugin": "macro_sync", "message": str(e)})

    async def detener(self):
        if self.plugin_db:
            await self.plugin_db.disconnect()
        self.logger.info("MacroProcessor detenido")