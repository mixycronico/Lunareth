#!/usr/bin/env python3
# tests/plugins/test_cli_manager.py
"""
test_cli_manager.py
Pruebas unitarias para el plugin cli_manager con validación de roles.
"""

import pytest
import asyncio
import json
import zstandard as zstd
from src.plugins.cli_manager.processors.cli_processor import CLIProcessor
from src.utils.config import load_secrets
from corec.entidad_base import Event

@pytest.mark.asyncio
async def test_cli_processor(monkeypatch):
    async def mock_get_all(self, pattern):
        return {
            "cli:market_data": json.dumps({"symbol": "BTC/USDT", "price": 70000.0}),
            "cli:macro_data": json.dumps({"dxy_price": 100.5, "dxy_change_percent": 0.0})
        }

    async def mock_connect(self):
        return True

    async def mock_responder_chat(self, message, context):
        return {"estado": "ok", "respuesta": f"Respuesta a: {message}"}

    async def mock_razonar(self, data, context):
        return {"estado": "ok", "respuesta": "Análisis de mercado simulado"}

    async def mock_regenerar_enjambre(self, canal, count):
        pass

    async def mock_connect_db(**kwargs):
        class MockConn:
            def cursor(self):
                class MockCursor:
                    def execute(self, query, params):
                        pass
                    def fetchone(self):
                        return ["superadmin"]
                    def close(self):
                        pass
                return MockCursor()
            def close(self):
                pass
        return MockConn()

    monkeypatch.setattr("redis.asyncio.Redis.get_all", mock_get_all)
    monkeypatch.setattr("src.plugins.cli_manager.utils.db.CLIDB.connect", mock_connect)
    monkeypatch.setattr("src.core.nucleus.CoreCNucleus.responder_chat", mock_responder_chat)
    monkeypatch.setattr("src.core.nucleus.CoreCNucleus.razonar", mock_razonar)
    monkeypatch.setattr("src.core.modules.registro.ModuloRegistro.regenerar_enjambre", mock_regenerar_enjambre)
    monkeypatch.setattr("psycopg2.connect", mock_connect_db)

    config = load_secrets("configs/plugins/cli_manager/cli_manager.yaml")
    processor = CLIProcessor(config, None, config.get("db_config"))
    await processor.inicializar(None)

    # Probar comando permitido
    event = Event(
        canal="cli_data",
        datos={"data": zstd.compress(json.dumps({"action": "monitor_dxy", "user_id": "user1"}).encode())},
        destino="cli_manager"
    )
    result = await processor.manejar_evento(event)
    assert "DXY: 100.50" in result["response"]

    # Probar comando restringido
    event = Event(
        canal="cli_data",
        datos={"data": zstd.compress(json.dumps({"action": "manage_swarm", "canal": "corec_stream_corec1", "count": 100, "user_id": "user1"}).encode())},
        destino="cli_manager"
    )
    result = await processor.manejar_evento(event)
    assert result["estado"] == "ok"  # superadmin tiene permiso

    await processor.detener()