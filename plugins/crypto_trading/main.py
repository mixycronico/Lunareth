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

            db_config = self.config.get("db_config", {
                "dbname": "crypto_trading_db",
                "user": "postgres",
                "password": "secure_password",
                "host": "localhost",
                "port": 5432
            })
            self.trading_db = TradingDB(db_config)
            await self.trading_db.connect()

            self.analyzer_processor = AnalyzerProcessor(config, self.redis_client)
            self.capital_processor = CapitalProcessor(config, self.redis_client, self.trading_db, self.nucleus)
            self.execution_processor = ExecutionProcessor({"open_trades": self.open_trades, "num_exchanges": len(self.config.get("exchanges", [])), "capital": self.capital}, self.redis_client)
            self.macro_processor = MacroProcessor(config, self.redis_client)
            self.monitor_processor = MonitorProcessor(config, self.redis_client, self.open_trades)
            self.predictor_processor = PredictorProcessor(config, self.redis_client)
            self.ia_processor = IAAnalisisProcessor(config, self.redis_client)
            self.strategy = MomentumStrategy(self.capital, self.ia_processor, self.predictor_processor)
            await self.predictor_processor.inicializar()
            await self.ia_processor.inicializar()

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

            self.exchange_processor = ExchangeProcessor(config, self.nucleus, self.strategy, self.execution_processor, self.settlement_processor, self.monitor_blocks)
            await self.exchange_processor.inicializar()

            activos = await self.exchange_processor.detectar_disponibles()
            for ex in activos:
                exchange_name = ex["name"]
                client = ex["client"]
                top_pairs = await self.exchange_processor.obtener_top_activos(client)
                self.trading_pairs_by_exchange[exchange_name] = top_pairs
                self.logger.info(f"Pares configurados para {exchange_name}: {top_pairs}")

            self.capital_processor.distribuir_capital(list(self.trading_pairs_by_exchange.keys()))
            self.settlement_processor = SettlementProcessor(config, self.redis_client, self.trading_db, self.nucleus, self.strategy, self.execution_processor, self.capital_processor)

            self.exchanges = list(self.trading_pairs_by_exchange.keys())
            for idx, exchange in enumerate(self.exchanges):
                asyncio.create_task(self.exchange_processor.monitor_exchange(exchange, self.trading_pairs_by_exchange[exchange], initial_offset=idx * 30))

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
            await self.nucleus.publicar_al
