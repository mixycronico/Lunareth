#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/test_comunicador_inteligente.py
Pruebas rigurosas para el plugin ComunicadorInteligente.
"""
import unittest
import asyncio
import json
from unittest.mock import AsyncMock, patch
from plugins.comunicador_inteligente.main import ComunicadorInteligente, RedNeuronalLigera, QLearningAgent

class TestComunicadorInteligente(unittest.TestCase):
    def setUp(self):
        self.config = {
            "nombre": "comunicador_inteligente",
            "canal": 4,
            "entidades": 10,
            "openrouter_api_key": "test_key",
            "training_log": "plugins/comunicador_inteligente/data/training.log",
            "redis_stream_input": "user_input_stream",
            "redis_stream_output": "user_output_stream"
        }
        self.nucleus = AsyncMock()
        self.nucleus.modulos = {"registro": AsyncMock()}
        self.nucleus.redis_config = {
            "host": "localhost",
            "port": 6379,
            "username": "test_user",
            "password": "test_password"
        }
        self.plugin = ComunicadorInteligente(self.nucleus, self.config)
        self.loop = asyncio.get_event_loop()

    async def test_inicializar(self):
        with patch("aioredis.from_url", AsyncMock()):
            await self.plugin.inicializar()
            self.assertIsNotNone(self.plugin.bloque)
            self.assertEqual(len(self.plugin.bloque.entidades), 10)
            self.nucleus.registrar_plugin.assert_called_with("comunicador_inteligente", self.plugin)

    async def test_procesar_mensaje_openrouter(self):
        with patch("aiohttp.ClientSession.post", AsyncMock(return_value=AsyncMock(json=AsyncMock(return_value={"choices": [{"text": "OK"}]})))):
            mensaje = {"texto": "Estado del sistema", "valor": 0.5}
            resultado = await self.plugin._procesar_mensaje(mensaje)
            self.assertIn("texto", resultado)
            self.assertEqual(resultado["texto"], "OK")

    async def test_procesar_mensaje_local(self):
        self.plugin.is_openrouter_available = False
        mensaje = {"texto": "Estado del sistema", "valor": 0.5}
        resultado = await self.plugin._procesar_mensaje(mensaje)
        self.assertIn("valor", resultado)
        self.assertIn("texto", resultado)

    async def test_entrenar_local(self):
        with open(self.config["training_log"], "w") as f:
            f.write(json.dumps({"entrada": {"valor": 0.5}, "salida": "OK"}) + "\n")
        await self.plugin.entrenar_local()
        self.assertTrue(hasattr(self.plugin.bayes_model, "classes_"))

    async def test_rl_decision(self):
        state = (0.5, 0)
        action = self.plugin.rl_model.choose_action(state)
        self.assertIn(action, ["responder", "analizar", "optimizar"])
        self.plugin.rl_model.update(state, action, 1.0, (0.6, 1))
        self.assertGreater(self.plugin.rl_model.q_table[str(state)][action], 0)

    async def test_escuchar_mensajes(self):
        with patch("aioredis.from_url", AsyncMock()) as mock_redis:
            mock_redis.return_value.xread.return_value = [
                ("user_input_stream", [(b"1-0", {"data": json.dumps({"texto": "Test", "valor": 0.5})})])
            ]
            with patch.object(self.plugin, "_procesar_mensaje", AsyncMock(return_value={"valor": 0.7, "texto": "OK"})):
                task = asyncio.create_task(self.plugin._escuchar_mensajes())
                await asyncio.sleep(0.1)
                task.cancel()
                mock_redis.return_value.xadd.assert_called()

    def test_all(self):
        async def run_tests():
            await self.test_inicializar()
            await self.test_procesar_mensaje_openrouter()
            await self.test_procesar_mensaje_local()
            await self.test_entrenar_local()
            await self.test_rl_decision()
            await self.test_escuchar_mensajes()
        self.loop.run_until_complete(run_tests())

if __name__ == "__main__":
    unittest.main()