#!/usr/bin/env python3
# tests/core/test_modulo_registro.py
"""
test_modulo_registro.py
Pruebas unitarias para el ModuloRegistro con soporte para millones de micro-celus.
"""

import pytest
import asyncio
from src.core.modules.registro import ModuloRegistro
from src.core.celu_entidad import CeluEntidadCoreC
from src.core.nucleus import CoreCNucleus

@pytest.mark.asyncio
async def test_modulo_registro(monkeypatch):
    async def mock_connect(**kwargs):
        class MockConn:
            def cursor(self):
                class MockCursor:
                    def execute(self, query, params):
                        pass
                    def fetchone(self):
                        return [0.4, 5]  # Carga, nodos activos
                    def close(self):
                        pass
                return MockCursor()
            def close(self):
                pass
        return MockConn()

    async def mock_redis(*args, **kwargs):
        class MockRedis:
            async def setex(self, key, ttl, value):
                pass
            async def get(self, key):
                return None
            async def delete(self, key):
                pass
            async def close(self):
                pass
        return MockRedis()

    monkeypatch.setattr("psycopg2.connect", mock_connect)
    monkeypatch.setattr("redis.asyncio.from_url", mock_redis)

    nucleus = CoreCNucleus(config_path="configs/core/corec_config_corec1.json", instance_id="corec1")
    modulo = ModuloRegistro()
    await modulo.inicializar(nucleus)

    # Simular alta carga
    modulo.load_threshold = 0.5
    modulo.enjambres["test_canal"] = [None] * 200000
    modulo.sub_enjambres["test_canal"] = [[None] * 100000, [None] * 100000]
    await modulo.optimize_swarms()
    assert len(modulo.enjambres["test_canal"]) <= 200000

    await modulo.detener()