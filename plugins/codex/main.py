#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/main.py
Plugin Codex con mejoras.
"""
import asyncio, logging
from corec.plugins.base import PluginBase
from processors.manager import CodexManager

class CodexPlugin(PluginBase):
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.config  = config["codex"]
        self.logger  = logging.getLogger("CodexPlugin")
        self.manager = CodexManager(nucleus, self.config)

    async def inicializar(self, nucleus, config):
        await self.manager.init()
        nucleus.comando_handlers["codex"] = self.manager.handle
        asyncio.create_task(self.manager.run_loop())
        self.logger.info("CodexPlugin inicializado con revise, métricas y seguridad")

    async def ejecutar(self):
        pass

    async def detener(self):
        await self.manager.teardown()
        self.logger.info("CodexPlugin detenido")

def inicializar(nucleus, config):
    plugin = CodexPlugin(nucleus, config)
    asyncio.create_task(plugin.inicializar(nucleus, config))
    return plugin