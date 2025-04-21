import logging
from typing import List, Dict, Any
from corec.core import ComponenteBase
import ccxt.async_support as ccxt
import asyncio

class ExchangeProcessor(ComponenteBase):
    def __init__(self, config, nucleus):
        self.config = config
        self.nucleus = nucleus
        self.logger = logging.getLogger("ExchangeProcessor")
        self.exchanges = self.config.get("exchanges", [])
        self.exchange_clients = {}
        self.api_limits = {}  # Rastrear límites de API por exchange
        self.liquidity_threshold = 100000  # Umbral de volumen en USD para liquidez

    async def inicializar(self):
        """Inicializa los clientes de los exchanges."""
        for ex in self.exchanges:
            try:
                nombre = ex["name"]
                exchange_class = getattr(ccxt, nombre)
                self.exchange_clients[nombre] = exchange_class({
                    "apiKey": ex["api_key"],
                    "secret": ex["api_secret"],
                    "enableRateLimit": True,
                })
                self.api_limits[nombre] = {"requests": 0, "last_reset": asyncio.get_event_loop().time()}
            except Exception as e:
                self.logger.warning(f"No se pudo inicializar cliente para {ex['name']}: {e}")

    async def _manage_api_limits(self, exchange_name: str):
        """Gestiona los límites de API para evitar excederlos."""
        current_time = asyncio.get_event_loop().time()
        limit = 1200  # Ejemplo: límite de Binance (1200 solicitudes por minuto)
        reset_interval = 60  # 1 minuto

        if current_time - self.api_limits[exchange_name]["last_reset"] > reset_interval:
            self.api_limits[exchange_name]["requests"] = 0
            self.api_limits[exchange_name]["last_reset"] = current_time

        self.api_limits[exchange_name]["requests"] += 1
        if self.api_limits[exchange_name]["requests"] > limit:
            wait_time = reset_interval - (current_time - self.api_limits[exchange_name]["last_reset"])
            self.logger.warning(f"Límite de API alcanzado para {exchange_name}, esperando {wait_time} segundos")
            await asyncio.sleep(wait_time)
            self.api_limits[exchange_name]["requests"] = 0
            self.api_limits[exchange_name]["last_reset"] = asyncio.get_event_loop().time()

    async def detectar_disponibles(self) -> List[Dict[str, Any]]:
        """Detecta exchanges activos y filtra los que no responden."""
        disponibles = []
        for ex in self.exchanges:
            try:
                nombre = ex["name"]
                client = self.exchange_clients.get(nombre)
                if client:
                    await self._manage_api_limits(nombre)
                    await client.fetch_status()
                    disponibles.append({"name": nombre, "client": client})
            except Exception as e:
                self.logger.warning(f"Exchange {ex['name']} no disponible: {e}")
        self.logger.info(f"Exchanges activos: {[e['name'] for e in disponibles]}")
        return disponibles

    async def obtener_top_activos(self, client) -> List[str]:
        """Retorna BTC, ETH y los 10 activos con mayor volumen por exchange, filtrando por liquidez."""
        try:
            await self._manage_api_limits(client.id)
            tickers = await client.fetch_tickers()
            # Filtrar por liquidez (volumen mínimo)
            filtered_tickers = {
                symbol: ticker for symbol, ticker in tickers.items()
                if "/" in symbol and float(ticker.get("quoteVolume", 0)) >= self.liquidity_threshold
            }
            ordenados = sorted(
                filtered_tickers.items(), key=lambda x: float(x[1].get("quoteVolume", 0)), reverse=True
            )
            top_activos = [k for k, _ in ordenados if "/" in k][:12]  # Top 12 para excluir BTC y ETH
            activos = ["BTC/USDT", "ETH/USDT"]
            for activo in top_activos:
                if activo not in ["BTC/USDT", "ETH/USDT"]:
                    activos.append(activo)
                if len(activos) >= 12:
                    break
            self.logger.info(f"Activos seleccionados para operar en {client.id}: {activos}")
            return activos
        except Exception as e:
            self.logger.error(f"Error al obtener top activos: {e}")
            # Reintento exponencial
            for attempt in range(3):
                await asyncio.sleep(2 ** attempt)
                try:
                    await self._manage_api_limits(client.id)
                    tickers = await client.fetch_tickers()
                    filtered_tickers = {
                        symbol: ticker for symbol, ticker in tickers.items()
                        if "/" in symbol and float(ticker.get("quoteVolume", 0)) >= self.liquidity_threshold
                    }
                    ordenados = sorted(
                        filtered_tickers.items(), key=lambda x: float(x[1].get("quoteVolume", 0)), reverse=True
                    )
                    top_activos = [k for k, _ in ordenados if "/" in k][:12]
                    activos = ["BTC/USDT", "ETH/USDT"]
                    for activo in top_activos:
                        if activo not in ["BTC/USDT", "ETH/USDT"]:
                            activos.append(activo)
                        if len(activos) >= 12:
                            break
                    self.logger.info(f"Activos seleccionados para operar en {client.id}: {activos}")
                    return activos
                except Exception as e:
                    self.logger.error(f"Reintento {attempt+1} fallido al obtener top activos: {e}")
            return ["BTC/USDT", "ETH/USDT"]

    async def close(self):
        for client in self.exchange_clients.values():
            await client.close()
