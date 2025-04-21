import logging
from typing import List, Dict, Any
from corec.core import ComponenteBase
import ccxt.async_support as ccxt
import asyncio
import json
import random
import heapq

class ExchangeProcessor(ComponenteBase):
    def __init__(self, config, nucleus, strategy, execution_processor, settlement_processor, monitor_blocks):
        self.config = config
        self.nucleus = nucleus
        self.strategy = strategy
        self.execution_processor = execution_processor
        self.settlement_processor = settlement_processor
        self.monitor_blocks = monitor_blocks
        self.logger = logging.getLogger("ExchangeProcessor")
        self.exchanges = self.config.get("exchanges", [])
        self.exchange_clients = {}
        self.api_limits = {}
        self.liquidity_threshold = 100000
        self.task_queues = {}  # Cola de prioridad por exchange

    async def inicializar(self):
        """Inicializa los clientes de los exchanges y comienza los bucles de monitoreo."""
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
                self.task_queues[nombre] = []
            except Exception as e:
                self.logger.warning(f"No se pudo inicializar cliente para {ex['name']}: {e}")

    async def _manage_api_limits(self, exchange_name: str):
        current_time = asyncio.get_event_loop().time()
        limit = 1200
        reset_interval = 60

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
            self.logger.error(f"Error al obtener top activos: {e}")
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

    async def monitor_exchange(self, exchange: str, pairs: list, initial_offset: float = 0):
        """Monitorea un exchange específico usando una cola de prioridad."""
        await asyncio.sleep(initial_offset)
        heapq.heappush(self.task_queues[exchange], (asyncio.get_event_loop().time(), "monitor"))

        while True:
            if not self.task_queues[exchange]:
                await asyncio.sleep(1)
                continue

            next_execution_time, task = heapq.heappop(self.task_queues[exchange])
            current_time = asyncio.get_event_loop().time()
            if next_execution_time > current_time:
                await asyncio.sleep(next_execution_time - current_time)

            try:
                now = datetime.datetime.now()
                start_time = now.replace(hour=6, minute=0, second=0, microsecond=0)
                end_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
                within_trading_hours = start_time <= now <= end_time

                avg_volatility = 0.01
                adjusted_interval = 180

                if within_trading_hours:
                    market_data = await self.redis.get("market_data")
                    if not market_data:
                        self.logger.warning(f"No hay datos de mercado disponibles para {exchange}")
                        heapq.heappush(self.task_queues[exchange], (asyncio.get_event_loop().time() + 180, "monitor"))
                        continue
                    market_data = json.loads(market_data)
                    macro_data = market_data["macro"]
                    crypto_data = market_data["crypto"]

                    adjustments = await self.redis.get("trading_flow_adjustments")
                    if adjustments:
                        adjustments = json.loads(adjustments)
                        interval_factor = adjustments.get("interval_factor", 1.0)
                        trade_multiplier_adjustment = adjustments.get("trade_multiplier", 2)
                        if adjustments.get("pause", False):
                            self.logger.info(f"Pausa por baja volatilidad para {exchange}, reprogramando en 300 segundos")
                            heapq.heappush(self.task_queues[exchange], (asyncio.get_event_loop().time() + 300, "monitor"))
                            continue
                    else:
                        interval_factor = 1.0
                        trade_multiplier_adjustment = 2

                    # Usar AnalyzerProcessor para analizar volatilidad y priorizar pares
                    volatility_result = await self.analyzer_processor.analizar_volatilidad(exchange, pairs)
                    if volatility_result["status"] != "ok":
                        heapq.heappush(self.task_queues[exchange], (asyncio.get_event_loop().time() + 180, "monitor"))
                        continue

                    prioritized_pairs = volatility_result["datos"]["prioritized_pairs"]
                    avg_volatility = volatility_result["datos"]["avg_volatility"]

                    for block in self.monitor_blocks:
                        result = await block.procesar(0.5)
                        if result["status"] != "success":
                            self.logger.warning(f"Error en monitoreo para {exchange}: {result['motivo']}")
                            continue

                        for pair, vol in prioritized_pairs:
                            combined_crypto_data = crypto_data.get(pair, {"volume": 0, "market_cap": 0})
                            prices = [50000 + i * 100 for i in range(50)]
                            sentiment = await self.strategy.calculate_momentum(macro_data, combined_crypto_data, prices, vol)
                            side = self.strategy.decide_trade(exchange, pair, sentiment, vol)

                            if side == "pending":
                                continue

                            trade_multiplier = self.strategy.get_trade_multiplier() * trade_multiplier_adjustment
                            async for trade_result in self.execution_processor.ejecutar_operacion(exchange, {
                                "precio": 50000,
                                "cantidad": 0.1,
                                "activo": pair,
                                "tipo": side
                            }, paper_mode=self.config.get("paper_mode", True), trade_multiplier=trade_multiplier):
                                self.open_trades[f"{exchange}:{pair}"] = trade_result
                                await self.settlement_processor.update_capital_after_trade(side, trade_result)

                base_interval = 180
                volatility_factor = max(avg_volatility / 0.01, 1.0)
                variation = random.uniform(-60, 60)
                adjusted_interval = max(120, min(240, (base_interval / volatility_factor + variation) * interval_factor))
                heapq.heappush(self.task_queues[exchange], (asyncio.get_event_loop().time() + adjusted_interval, "monitor"))
                self.logger.info(f"[ExchangeProcessor] Próximo monitoreo para {exchange} en {adjusted_interval} segundos (Volatilidad promedio: {avg_volatility})")
            except Exception as e:
                self.logger.error(f"[ExchangeProcessor] Error en bucle de monitoreo para {exchange}: {e}")
                heapq.heappush(self.task_queues[exchange], (asyncio.get_event_loop().time() + 180, "monitor"))

    async def close(self):
        for client in self.exchange_clients.values():
            await client.close()
