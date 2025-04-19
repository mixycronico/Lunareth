#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/main.py
Orquesta el plugin CryptoTrading.
"""
import asyncio
import logging
import json
from corec.core import ComponenteBase
from .processors.exchange_processor import ExchangeProcessor
from .processors.capital_processor import CapitalProcessor
from .processors.settlement_processor import SettlementProcessor
from .processors.macro_processor import MacroProcessor
from .processors.monitor_processor import MonitorProcessor
from .processors.predictor_processor import PredictorProcessor
from .processors.analyzer_processor import AnalyzerProcessor
from .processors.execution_processor import ExecutionProcessor
from .processors.user_processor import UserProcessor

class CryptoTrading(ComponenteBase):
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.config = config
        self.logger = logging.getLogger("CryptoTrading")
        self.redis_client = nucleus.redis_client
        self.components = []

    async def inicializar(self):
        self.components.append(ExchangeProcessor(self.config, self.redis_client))
        self.components.append(CapitalProcessor(self.config, self.redis_client))
        self.components.append(SettlementProcessor(self.config, self.redis_client))
        self.components.append(MacroProcessor(self.config, self.redis_client))
        self.components.append(MonitorProcessor(self.config, self.redis_client))
        self.components.append(PredictorProcessor(self.config, self.redis_client))
        self.components.append(AnalyzerProcessor(self.config, self.redis_client))
        self.components.append(ExecutionProcessor(self.config, self.redis_client))
        self.components.append(UserProcessor(self.config, self.redis_client))
        for component in self.components:
            await component.inicializar()
        self.logger.info("CryptoTrading inicializado")

    async def ejecutar(self):
        tasks = [component.ejecutar() for component in self.components]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def detener(self):
        for component in self.components:
            await component.detener()
        self.logger.info("CryptoTrading detenido")

def inicializar(nucleus, config):
    plugin = CryptoTrading(nucleus, config)
    asyncio.create_task(plugin.inicializar())