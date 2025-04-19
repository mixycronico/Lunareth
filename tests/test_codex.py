#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/tests/test_codex.py
Pruebas unitarias para el plugin Codex.
"""
import unittest
import asyncio
import os
from unittest.mock import AsyncMock, patch
from plugins.codex.processors.manager import CodexManager
from plugins.codex.processors.reviser import CodexReviser
from plugins.codex.processors.generator import CodexGenerator
from plugins.codex.processors.memory import CodexMemory
from corec.core import aioredis

class TestCodex(unittest.TestCase):
    def setUp(self):
        self.config = {
            "intervalo_revision": 300,
            "directorio_objetivo": "plugins/",
            "exclude_patterns": ["tests/*"],
            "languages": ["python"],
            "max_file_size": 1000000,
            "circuit_breaker": {"max_failures": 2, "reset_timeout": 60},
            "website_output_dir": "generated_websites/",
            "plugin_output_dir": "plugins/",
            "website_templates": {"react": "utils/templates/react_app"},
            "plugin_templates": {"corec_plugin": "utils/templates/plugin"}
        }
        self.redis_client = AsyncMock()
        self.nucleus = biblioteka = AsyncMock(redis_client=self.redis_client)
        self.manager = CodexManager(self.nucleus, self.config)
        self.loop = asyncio.get_event_loop()

    async def test_manager_inicializar(self):
        with patch("os.makedirs", AsyncMock()):
            await self.manager.inicializar()
            self.assertIsNotNone(self.manager.reviser)
            self.assertIsNotNone(self.manager.generator)
            self.assertIsNotNone(self.manager.memory)

    async def test_reviser_python(self):
        reviser = CodexReviser(self.config)
        codigo = "def test():\n    pass"
        nuevo = await reviser.revisar_codigo(codigo, "test.py")
        self.assertIn("def test", nuevo)
        self.assertNotIn("pass", nuevo)

    async def test_generator_website(self):
        generator = CodexGenerator(self.config)
        with patch("shutil.copytree", AsyncMock()):
            result = await generator.generar_website({"template": "react", "project_name": "test_web"})
            self.assertEqual(result["status"], "ok")
            self.assertIn("generated_websites/test_web", result["output_dir"])

    async def test_generator_plugin(self):
        generator = CodexGenerator(self.config)
        with patch("shutil.copytree", AsyncMock()):
            result = await generator.generar_plugin({"plugin_name": "test_plugin"})
            self.assertEqual(result["status"], "ok")
            self.assertIn("plugins/test_plugin", result["output_dir"])

    async def test_memory_revision(self):
        memory = CodexMemory(self.redis_client)
        contenido = "def test(): pass"
        with patch.object(self.redis_client, "get", AsyncMock(return_value=None)):
            self.assertTrue(await memory.necesita_revision("test.py", contenido))

    def test_all(self):
        async def run_tests():
            await self.test_manager_inicializar()
            await self.test_reviser_python()
            await self.test_generator_website()
            await self.test_generator_plugin()
            await self.test_memory_revision()
        self.loop.run_until_complete(run_tests())

if __name__ == "__main__":
    unittest.main()