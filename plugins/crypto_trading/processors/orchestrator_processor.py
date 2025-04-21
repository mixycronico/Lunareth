import asyncio
import logging
import datetime  # Añadimos el import
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
from plugins.crypto_trading.scheduler import Scheduler

class OrchestratorProcessor(ComponenteBase):
    def __init__(self, config, nucleus, redis_client):
        self.config = config
        self.nucleus = nucleus
        self.redis_client = redis_client
        self.logger = logging.getLogger("OrchestratorProcessor")
        self.trading_pairs_by_exchange = {}
        self.trading_blocks = []
        self.monitor_blocks = []
        self.processors = {}
        self.trading_db = None
        self.paper_mode = True
        self.scheduler = None

    async def initialize(self):
        """Inicializa todos los componentes y programa las tareas."""
        try:
            self.capital = self.config.get("total_capital", 100)
            self.paper_mode = self.config.get("paper_mode", True)
            self.logger.info(f"[OrchestratorProcessor] Modo paper: {self.paper_mode}")

            db_config = self.config.get("db_config", {
                "dbname": "crypto_trading_db",
                "user": "postgres",
                "password": "secure_password",
                "host": "localhost",
                "port": 5432
            })
            self.trading_db = TradingDB(db_config)
            await self.trading_db.connect()

            # Inicializar procesadores dinámicamente
            await self.load_processor("analyzer", AnalyzerProcessor(self.config["analyzer_config"], self.redis_client))
            await self.load_processor("capital", CapitalProcessor(self.config["capital_config"], self.redis_client, self.trading_db, self.nucleus))
            await self.load_processor("execution", ExecutionProcessor({"open_trades": {}, "num_exchanges": len(self.config.get("exchange_config", {}).get("exchanges", [])), "capital": self.capital}, self.redis_client))
            await self.load_processor("macro", MacroProcessor(self.config["macro_config"], self.redis_client))
            await self.load_processor("monitor", MonitorProcessor(self.config["monitor_config"], self.redis_client, self.processors["execution"].open_trades))
            await self.load_processor("predictor", PredictorProcessor(self.config["predictor_config"], self.redis_client))
            await self.load_processor("ia", IAAnalisisProcessor(self.config, self.redis_client))
            await self.load_processor("strategy", MomentumStrategy(self.capital, self.processors["ia"], self.processors["predictor"]))
            await self.processors["predictor"].inicializar()
            await self.processors["ia"].inicializar()

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
                execution_processor=self.processors["execution"],
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
                analyzer_processor=self.processors["analyzer"],
                monitor_processor=self.processors["monitor"]
            )
            self.monitor_blocks.append(monitor_block)

            self.nucleus.bloques.extend(self.trading_blocks + self.monitor_blocks)

            await self.load_processor("exchange", ExchangeProcessor(self.config["exchange_config"], self.nucleus, self.processors["strategy"], self.processors["execution"], self.processors.get("settlement"), self.monitor_blocks))
            await self.processors["exchange"].inicializar()

            activos = await self.processors["exchange"].detectar_disponibles()
            for ex in activos:
                exchange_name = ex["name"]
                client = ex["client"]
                top_pairs = await self.processors["exchange"].obtener_top_activos(client)
                self.trading_pairs_by_exchange[exchange_name] = top_pairs
                self.logger.info(f"Pares configurados para {exchange_name}: {top_pairs}")

            self.processors["capital"].distribuir_capital(list(self.trading_pairs_by_exchange.keys()))
            await self.load_processor("settlement", SettlementProcessor(self.config["settlement_config"], self.redis_client, self.trading_db, self.nucleus, self.processors["strategy"], self.processors["execution"], self.processors["capital"]))
            await self.processors["settlement"].restore_state()

            self.exchanges = list(self.trading_pairs_by_exchange.keys())

            # Inicializar el scheduler
            self.scheduler = Scheduler()
            self.scheduler.start()

            # Programar tareas con el scheduler
            # 1) Datos macro periódicos
            self.scheduler.schedule_periodic(
                func=self.processors["macro"].fetch_and_publish_data,
                seconds=self.config["macro_config"]["update_interval"],
                job_id="fetch_macro"
            )

            # 2) Monitoreo de cada exchange con offset
            for idx, exchange in enumerate(self.exchanges):
                self.scheduler.schedule_periodic(
                    func=lambda ex=exchange: asyncio.create_task(
                        self.processors["exchange"].monitor_exchange(
                            ex, self.trading_pairs_by_exchange[ex]
                        )
                    ),
                    seconds=self.config["monitor_config"]["update_interval"],
                    job_id=f"monitor_{exchange}",
                    start_delay=idx * 30
                )

            # 3) Cierre diario según settlement_time
            h, m = map(int, self.config["settlement_config"]["settlement_time"].split(":"))
            if not (0 <= h <= 23 and 0 <= m <= 59):
                raise ValueError("settlement_time debe estar en formato HH:MM y dentro de rangos válidos (00:00-23:59)")
            self.scheduler.schedule_cron(
                func=self.processors["settlement"].daily_close_process,
                minute=m,
                hour=h,
                job_id="daily_close"
            )

            # Otras tareas periódicas
            self.scheduler.schedule_periodic(
                func=self.processors["predictor"].adjust_trading_flow,
                seconds=300,
                job_id="adjust_trading_flow"
            )
            self.scheduler.schedule_periodic(
                func=self.processors["monitor"].continuous_open_trades_monitor,
                seconds=30,
                job_id="continuous_monitor",
                args=[self.processors["settlement"].close_trade]
            )
            self.scheduler.schedule_periodic(
                func=self.processors["settlement"].micro_cycle_loop,
                seconds=60,
                job_id="micro_cycle"
            )
            self.scheduler.schedule_periodic(
                func=self.processors["settlement"].handle_market_crash,
                seconds=60,
                job_id="market_crash"
            )
            self.scheduler.schedule_periodic(
                func=self.processors["settlement"].monitor_services,
                seconds=300,
                job_id="monitor_services"
            )
            self.scheduler.schedule_periodic(
                func=self.processors["capital"].adjust_base_capital,
                seconds=86400,
                job_id="adjust_base_capital"
            )
            self.scheduler.schedule_periodic(
                func=self.processors["capital"].vote_strategy,
                seconds=86400,
                job_id="vote_strategy"
            )

            self.logger.info("[OrchestratorProcessor] Componentes inicializados y tareas programadas con APScheduler")
        except Exception as e:
            self.logger.error(f"[OrchestratorProcessor] Error al inicializar: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion_plugin",
                "plugin_id": "crypto_trading",
                "mensaje": str(e),
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            raise

    async def load_processor(self, name: str, processor: ComponenteBase):
        """Carga un procesador dinámicamente."""
        self.processors[name] = processor
        self.logger.info(f"Procesador {name} cargado dinámicamente")

    async def unload_processor(self, name: str):
        """Descarga un procesador dinámicamente."""
        if name in self.processors:
            processor = self.processors[name]
            if hasattr(processor, "detener"):
                await processor.detener()
            del self.processors[name]
            self.logger.info(f"Procesador {name} descargado dinámicamente")

    async def detener(self):
        for name in list(self.processors.keys()):
            await self.unload_processor(name)
        self.logger.info("[OrchestratorProcessor] Componentes detenidos")
