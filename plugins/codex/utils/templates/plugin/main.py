#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/{{ plugin_name }}/main.py
Plugin {{ plugin_name | capitalize }} para CoreC.
"""
import asyncio
import logging
from corec.core import ComponenteBase

class {{ plugin_name | capitalize }}Plugin(ComponenteBase):
    def __init__(self, nucleus, config):
        super().__init__()
        self.nucleus = nucleus
        self.config = config.get("{{ plugin_name }}", {})
        self.logger = logging.getLogger("{{ plugin_name | capitalize }}Plugin")

    async def inicializar(self):
        self.logger.info("{{ plugin_name | capitalize }}Plugin inicializado")

    async def ejecutar(self):
        self.logger.info("Ejecutando {{ plugin_name | capitalize }}Plugin")

    async def detener(self):
        self.logger.info("{{ plugin_name | capitalize }}Plugin detenido")

def inicializar(nucleus, config):
    plugin = {{ plugin_name | capitalize }}Plugin(nucleus, config)
    asyncio.create_task(plugin.inicializar())
    return plugin