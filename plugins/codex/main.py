#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/main.py
Plugin Codex para CoreC – optimiza código, genera websites y plugins.
"""
import asyncio
import logging
from corec.core import ComponenteBase
from plugins.codex.processors.manager import CodexManager

class CodexPlugin(ComponenteBase):
    def __init__(self, nucleus, config):
        super().__init__()
        self.nucleus = nucleus
        self.config = config.get("codex", {})
        self.logger = logging.getLogger("CodexPlugin")
        self.manager = CodexManager(nucleus, self.config)

    async def inicializar(self):
        await self.manager.inicializar()
        self.logger.info("CodexPlugin inicializado")
        asyncio.create_task(self.manager.ejecutar())

    async def ejecutar(self):
        await self.manager.ejecutar()

    async def detener(self):
        await self.manager.detener()
        self.logger.info("CodexPlugin detenido")

def inicializar(nucleus, config):
    plugin = CodexPlugin(nucleus, config)
    asyncio.create_task(plugin.inicializar())
    return plugin