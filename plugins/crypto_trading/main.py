import asyncio
import logging
import yaml
from typing import Dict, Any
from corec.core import ComponenteBase
from corec.entities import crear_entidad
from plugins.crypto_trading.blocks.trading_block import TradingBlock
from plugins.crypto_trading.blocks.monitor_block import MonitorBlock
from plugins.crypto_trading.processors.analyzer_processor import AnalyzerProcessor
from plugins.crypto_trading.processors.capital_processor import CapitalProcessor
from plugins.crypto_trading.processors.exchange_processor import ExchangeProcessor
from plugins.crypto_trading.processors.execution_processor import ExecutionProcessor
from plugins.crypto_trading.processors.macro_processor import MacroProcessor
from plugins.crypto_trading.processors.monitor_processor import MonitorProcessor
from plugins.crypto_trading.processors.predictor_processor import PredictorProcessor
from plugins.crypto_trading.processors.ia_analysis_processor import IAAnalisisProcessor
from plugins.crypto_trading.processors.settlement_processor import SettlementProcessor
from plugins.crypto_trading.strategies.momentum_strategy import MomentumStrategy
from plugins.crypto_trading.utils.db import TradingDB
import datetime
import random
import json
import heapq

class CryptoTrading(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("CryptoTrading")
        self.nucleus = None
        self.redis_client = None
        self.trading_pairs_by_exchange = {}
        self.trading_blocks = []
        self.monitor_blocks = []
        self.analyzer_processor = None
        self.capital_processor = None
        self.exchange_processor = None
        self.execution_processor = None
        self.macro_processor = None
        self.monitor_processor = None
        self.predictor_processor = None
        self.ia_processor = None
        self.settlement_processor = None
        self.strategy = None
        self.trading_db = None
        self.paper_mode = True
        self.open_trades = {}
        self.config = None
        self.task_queue = []  # Usar una cola de prioridad para escalonamiento

    async def inicializar(self, nucleus, config=None):
        try:
            self.nucleus = nucleus
            self.redis_client = self.nucleus.redis_client
            if not self.redis_client:
                raise ValueError("Redis client no inicializado")

            self.config = config
            self.capital = config.get("total_capital", 100)

            self.paper_mode = config.get("paper_mode", True)
            self.logger.info(f"[CryptoTrading] Modo paper: {self.paper_mode}")

            self.analyzer_processor = AnalyzerProcessor(config, self.redis_client)
            self.capital_processor = CapitalProcessor(config, self.redis_client, self.trading_db, self.nucleus)
            self.exchange_processor = ExchangeProcessor(config, self.nucleus)
            await self.exchange_processor.inicializar()
            self.execution_processor = ExecutionProcessor({"open_trades": self.open_trades, "num_exchanges": len(self.exchange_processor.exchanges), "capital": self.capital}, self.redis_client)
            self.macro_processor = MacroProcessor(config, self.redis_client)
            self.monitor_processor = MonitorProcessor(config, self.redis_client, self.open_trades)
            self.predictor_processor = PredictorProcessor(config, self.redis_client)
            self.ia_processor = IAAnalisisProcessor(config, self.redis_client)
            self.strategy = MomentumStrategy(self.capital, self.ia_processor, self.predictor_processor)
            self.settlement_processor = SettlementProcessor(config, self.redis_client, self.trading_db, self.nucleus, self.strategy, self.execution_processor, self.capital_processor)
            await self.predictor_processor.inicializar()
            await self.ia_processor.inicializar()

            db_config = self.config.get("db_config", {
                "dbname": "crypto_trading_db",
                "user": "postgres",
                "password": "secure_password",
                "host": "localhost",
                "port": 5432
            })
            self.trading_db = TradingDB(db_config)
            await self.trading_db.connect()

            activos = await self.exchange_processor.detectar_disponibles()
            for ex in activos:
                exchange_name = ex["name"]
                client = ex["client"]
                top_pairs = await self.exchange_processor.obtener_top_activos(client)
                self.trading_pairs_by_exchange[exchange_name] = top_pairs
                self.logger.info(f"Pares configurados para {exchange_name}: {top_pairs}")

            self.capital_processor.distribuir_capital(list(self.trading_pairs_by_exchange.keys()))

            trading_entities = [
                crear_entidad(f"trade_ent_{i}", 3, lambda carga: {"valor": 0.5})
                for i in range(1000)
            ]
            trading_block = TradingBlock(
                id="trading_block_1",
                canal=3,
                entidades=trading_entities,
                max_size_mb=5,
                nucleus=self.nucleus,
                execution_processor=self.execution_processor,
                trading_db=self.trading_db
            )
            self.trading_blocks.append(trading_block)

            monitor_entities = [
                crear_entidad(f"monitor_ent_{i}", 3, lambda carga: {"valor": 0.5})
                for i in range(1000)
            ]
            monitor_block = MonitorBlock(
                id="monitor_block_1",
                canal=3,
                entidades=monitor_entities,
                max_size_mb=5,
                nucleus=self.nucleus,
                analyzer_processor=self.analyzer_processor,
                monitor_processor=self.monitor_processor
            )
            self.monitor_blocks.append(monitor_block)

            self.nucleus.bloques.extend(self.trading_blocks + self.monitor_blocks)

            self.exchanges = list(self.trading_pairs_by_exchange.keys())
            for idx, exchange in enumerate(self.exchanges):
                # Programar la primera ejecución con desfase
                heapq.heappush(self.task_queue, (idx * 30, exchange))

            asyncio.create_task(self._monitor_queue())
            asyncio.create_task(self.macro_processor.data_fetch_loop())
            asyncio.create_task(self.predictor_processor.adjust_trading_flow())
            asyncio.create_task(self.monitor_processor.continuous_open_trades_monitor(self.settlement_processor.close_trade))
            asyncio.create_task(self.settlement_processor.daily_close_loop(self.open_trades))
            asyncio.create_task(self.settlement_processor.micro_cycle_loop(self.open_trades))
            asyncio.create_task(self.settlement_processor.handle_market_crash(self.open_trades))
            asyncio.create_task(self.capital_processor.adjust_base_capital())

            self.logger.info("[CryptoTrading] Plugin inicializado correctamente")
        except Exception as e:
            self.logger.error(f"[CryptoTrading] Error al inicializar: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion_plugin",
                "plugin_id": "crypto_trading",
                "mensaje": str(e),
                "timestamp": datetime.datetime.utcnow().timestamp()
            })
            raise

    async def manejar_comando(self, comando: Dict[str, Any]) -> Dict[str, Any]:
        try:
            action = comando.get("action")
            params = comando.get("params", {})

            if action == "ejecutar_operacion":
                return await self._execute_trade(params.get("exchange"), params.get("pair"), params.get("side"))
            else:
                return {"status": "error", "message": f"Acción no soportada: {action}"}
        except Exception as e:
            self.logger.error(f"[CryptoTrading] Error al manejar comando: {e}")
            return {"status": "error", "message": str(e)}

    async def _monitor_queue(self):
        """Procesa la cola de prioridad para manejar el monitoreo de exchanges."""
        while True:
            if not self.task_queue:
                await asyncio.sleep(1)
                continue

            next_execution_time, exchange = heapq.heappop(self.task_queue)
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
                    market_data = await self.redis_client.get("market_data")
                    if not market_data:
                        self.logger.warning(f"No hay datos de mercado disponibles para {exchange}")
                        heapq.heappush(self.task_queue, (asyncio.get_event_loop().time() + 180, exchange))
                        continue
                    market_data = json.loads(market_data)
                    macro_data = market_data["macro"]
                    crypto_data = market_data["crypto"]

                    adjustments = await self.redis_client.get("trading_flow_adjustments")
                    if adjustments:
                        adjustments = json.loads(adjustments)
                        interval_factor = adjustments.get("interval_factor", 1.0)
                        trade_multiplier_adjustment = adjustments.get("trade_multiplier", 2)
                        if adjustments.get("pause", False):
                            self.logger.info(f"Pausa por baja volatilidad para {exchange}, reprogramando en 300 segundos")
                            heapq.heappush(self.task_queue, (asyncio.get_event_loop().time() + 300, exchange))
                            continue
                    else:
                        interval_factor = 1.0
                        trade_multiplier_adjustment = 2

                    # Usar AnalyzerProcessor para analizar volatilidad
                    volatility_result = await self.analyzer_processor.analizar_volatilidad()
                    if volatility_result["status"] == "ok":
                        volatilidad = volatility_result["datos"]
                        prioritized_pairs = []
                        pairs = self.trading_pairs_by_exchange[exchange]
                        for pair in pairs:
                            for v in volatilidad:
                                if v["symbol"] == pair and v["alerta"]:
                                    prioritized_pairs.append((pair, v["volatilidad"]))
                                    break
                        for pair in pairs:
                            if not any(p[0] == pair for p in prioritized_pairs):
                                prioritized_pairs.append((pair, 0.01))

                        avg_volatility = sum(v for _, v in prioritized_pairs) / len(prioritized_pairs) if prioritized_pairs else 0.01

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
                                }, paper_mode=self.paper_mode, trade_multiplier=trade_multiplier):
                                    self.open_trades[f"{exchange}:{pair}"] = trade_result
                                    await self.settlement_processor.update_capital_after_trade(side, trade_result)

                base_interval = 180
                volatility_factor = max(avg_volatility / 0.01, 1.0)
                variation = random.uniform(-60, 60)
                adjusted_interval = max(120, min(240, (base_interval / volatility_factor + variation) * interval_factor))
                heapq.heappush(self.task_queue, (asyncio.get_event_loop().time() + adjusted_interval, exchange))
                self.logger.info(f"[CryptoTrading] Próximo monitoreo para {exchange} en {adjusted_interval} segundos (Volatilidad promedio: {avg_volatility})")
            except Exception as e:
                self.logger.error(f"[CryptoTrading] Error en bucle de monitoreo para {exchange}: {e}")
                heapq.heappush(self.task_queue, (asyncio.get_event_loop().time() + 180, exchange))

    async def detener(self):
        await self.exchange_processor.close()
        await self.trading_db.disconnect()
        self.logger.info("[CryptoTrading] Plugin detenido")
