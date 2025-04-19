---

### **Pruebas: `tests/test_interface_system.py`**
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_interface_system.py
Pruebas rigurosas para el plugin InterfaceSystem.
"""
import unittest
import asyncio
import json
from unittest.mock import AsyncMock, patch
from plugins.interface_system.main import InterfaceSystem
from plugins.interface_system.brain import JarvisBrain
from plugins.interface_system.controller import InterfaceController

class TestInterfaceSystem(unittest.TestCase):
    def setUp(self):
        self.config = {
            "nombre": "interface_system",
            "canal": 8,
            "entidades": 10,
            "use_redis": False,
            "redis_stream_input": "user_input_stream",
            "redis_stream_output": "user_output_stream"
        }
        self.nucleus = AsyncMock()
        self.nucleus.modulos = {"registro": AsyncMock(), "auditoria": AsyncMock()}
        self.nucleus.plugins = {"comunicador_inteligente": AsyncMock()}
        self.nucleus.redis_config = {
            "host": "localhost",
            "port": 6379,
            "username": "test_user",
            "password": "test_password"
        }
        self.plugin = InterfaceSystem(self.nucleus, self.config)
        self.loop = asyncio.get_event_loop()

    async def test_inicializar(self):
        with patch("aioredis.from_url", AsyncMock()):
            await self.plugin.inicializar()
            self.assertIsNotNone(self.plugin.bloque)
            self.assertEqual(len(self.plugin.bloque.entidades), 10)
            self.nucleus.registrar_plugin.assert_called_with("interface_system", self.plugin)

    async def test_procesar_comando_status(self):
        with patch.object(self.plugin.controller, "estado_sistema", AsyncMock(return_value={
            "modulos_activos": ["registro"],
            "plugins_activos": ["comunicador_inteligente"],
            "bloques": [{"id": "b1", "canal": 1, "fitness": 0.9, "entidades": 100}],
            "alertas": [],
            "nodos": 1
        })):
            resultado = await self.plugin._procesar_comando({"comando": "status"})
            self.assertEqual(resultado["texto"], "Estado mostrado correctamente 🌱")

    async def test_procesar_comando_chat(self):
        with patch.object(self.plugin.controller, "enviar_chat", AsyncMock(return_value="OK")):
            resultado = await self.plugin._procesar_comando({"comando": "chat Test"})
            self.assertEqual(resultado["texto"], "OK")

    async def test_brain_memoria(self):
        self.plugin.brain.recordar("test", "respuesta")
        self.assertEqual(len(self.plugin.brain.mem["conversacion"]), 1)
        self.assertEqual(self.plugin.brain.mem["conversacion"][0]["q"], "test")

    def test_all(self):
        async def run_tests():
            await self.test_inicializar()
            await self.test_procesar_comando_status()
            await self.test_procesar_comando_chat()
            await self.test_brain_memoria()
        self.loop.run_until_complete(run_tests())

if __name__ == "__main__":
    unittest.main()