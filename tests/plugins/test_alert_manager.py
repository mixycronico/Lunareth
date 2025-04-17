#!/usr/bin/env python3
# tests/plugins/test_alert_manager.py
"""
test_alert_manager.py
Pruebas unitarias para el plugin alert_manager con notificaciones por email (SendGrid) y Discord.
"""

import pytest
import asyncio
import json
import zstandard as zstd
from src.plugins.alert_manager.processors.alert_processor import AlertProcessor
from src.utils.config import load_secrets
from corec.entidad_base import Event

@pytest.mark.asyncio
async def test_alert_processor(monkeypatch):
    async def mock_connect(self):
        return True

    async def mock_xadd(self, stream, data):
        pass

    async def mock_sendgrid(*args, **kwargs):
        class MockClient:
            def send(self, mail):
                class MockResponse:
                    status_code = 202
                return MockResponse()
        return MockClient()

    async def mock_post(*args, **kwargs):
        class MockResponse:
            status = 204
        return MockResponse()

    monkeypatch.setattr("src.plugins.alert_manager.utils.db.AlertDB.connect", mock_connect)
    monkeypatch.setattr("redis.asyncio.Redis.xadd", mock_xadd)
    monkeypatch.setattr("sendgrid.SendGridAPIClient", mock_sendgrid)
    monkeypatch.setattr("aiohttp.ClientSession.post", mock_post)

    config = load_secrets("configs/plugins/alert_manager/alert_manager.yaml")
    processor = AlertProcessor(config, None, config.get("db_config"))
    await processor.inicializar(None)

    event = Event(
        canal="alertas",
        datos={"data": zstd.compress(json.dumps({"tipo": "dxy_change", "message": "DXY cambió 1.2%"}).encode())},
        destino="alert_manager"
    )
    await processor.manejar_evento(event)

    result = await processor.procesar_alerta({"tipo": "dxy_change", "message": "DXY cambió 1.2%"})
    assert result["estado"] == "ok"
    assert result["alerta_id"].startswith("alert_")

    await processor.detener()