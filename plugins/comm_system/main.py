#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/comm_system/main.py
Plugin CommSystem para CoreC: chat real, memoria y generación on‑demand.
"""
import asyncio
import logging
from corec.plugins.base import PluginBase
from processors.manager import CommManager

class CommSystemPlugin(PluginBase):
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.config  = config["comm_system"]
        self.logger  = logging.getLogger("CommSystem")
        self.manager = CommManager(nucleus, self.config)

    async def inicializar(self, nucleus, config):
        # Inicializa Redis + memoria + AI
        await self.manager.init_redis()
        # Registra el handler de comandos
        nucleus.comando_handlers["comm_system"] = self.manager.handle_command
        # Arranca el loop de escucha de comandos
        asyncio.create_task(self.manager.run_loop())
        self.logger.info("CommSystemPlugin inicializado")

    async def ejecutar(self):
        # Toda la lógica está en el manager
        pass

    async def detener(self):
        await self.manager.teardown()
        self.logger.info("CommSystemPlugin detenido")

def inicializar(nucleus, config):
    plugin = CommSystemPlugin(nucleus, config)
    asyncio.create_task(plugin.inicializar(nucleus, config))
    return plugin