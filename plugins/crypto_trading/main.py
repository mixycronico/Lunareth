#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/main.py
Inicialización del plugin CryptoTrading con CoreC.
"""

import asyncio
import logging
from corec.plugins.base import PluginBase
from plugins.crypto_trading.processors.analyzer_processor import AnalyzerProcessor
from plugins.crypto_trading.processors.execution_processor import ExecutionProcessor
from plugins.crypto_trading.processors.monitor_processor import MonitorProcessor
from plugins.crypto_trading.processors.macro_processor import MacroProcessor
from plugins.crypto_trading.processors.predictor_processor import PredictorProcessor
from plugins.crypto_trading.processors.settlement_processor import SettlementProcessor
from plugins.crypto_trading.utils.db import TradingDB

class CryptoTrading(PluginBase):
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.config = config["crypto_trading"]
        self.logger = logging.getLogger("CryptoTrading")
        self.db = TradingDB(self.config["db_config"])

        self.analyzer = AnalyzerProcessor(self.config, self.db, nucleus.redis_client)
        self.execution = ExecutionProcessor(self.config, self.db, nucleus.redis_client)
        self.monitor   = MonitorProcessor(self.config, self.db, nucleus.redis_client)
        self.macro     = MacroProcessor(self.config, self.db, nucleus.redis_client)
        self.predictor = PredictorProcessor(self.config, self.db, nucleus.redis_client)
        self.settlement= SettlementProcessor(self.config, self.db, nucleus.redis_client)

    async def inicializar(self, nucleus, config):
        self.logger.info("Inicializando CryptoTrading...")
        await self.db.connect()

        await self.analyzer.inicializar()
        await self.execution.inicializar()
        await self.monitor.inicializar()
        await self.macro.inicializar()
        await self.predictor.inicializar()
        await self.settlement.inicializar()

        self.logger.info("CryptoTrading listo")

    async def ejecutar(self):
        self.logger.info("Ejecutando CryptoTrading...")
        await asyncio.gather(
            self.analyzer.ejecutar(),
            self.execution.ejecutar(),
            self.monitor.ejecutar(),
            self.macro.ejecutar(),
            self.predictor.ejecutar(),
            self.settlement.ejecutar()
        )

    async def detener(self):
        self.logger.info("Deteniendo CryptoTrading...")
        await self.analyzer.detener()
        await self.execution.detener()
        await self.monitor.detener()
        await self.macro.detener()
        await self.predictor.detener()
        await self.settlement.detener()
        await self.db.disconnect()
        self.logger.info("CryptoTrading detenido")

def inicializar(nucleus, config):
    plugin = CryptoTrading(nucleus, config)
    asyncio.create_task(plugin.inicializar(nucleus, config))
    return plugin