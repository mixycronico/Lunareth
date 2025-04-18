#!/usr/bin/env python3
# tests/plugins/test_system_analyzer.py
"""
test_system_analyzer.py
Pruebas unitarias para el plugin system_analyzer.
"""

import pytest
import asyncio
import json
import zstandard as zstd
from src.plugins.system_analyzer.processors.analyzer_processor import AnalyzerProcessor
from src.utils.config import load_secrets
from corec.entidad_base import Event

@pytest.mark.asyncio
async def test_analyzer_processor(monkeypatch):
    async def mock_xadd(self, stream, data):
        pass

    async def mock_connect(self):
        return True

    async def mock_razonar(self, data, context):
        return {"estado": "ok", "respuesta": "Análisis simulado"}

    monkeypatch.setattr("redis.asyncio.Redis.xadd", mock_xadd)
    monkeypatch.setattr("src.plugins.system_analyzer.utils.db.AnalyzerDB.connect", mock_connect)
    monkeypatch.setattr("src.core.nucleus.CoreCNucleus.razonar", mock_razonar)

    config = load_secrets("configs/plugins/system_analyzer/system_analyzer.yaml")
    processor = AnalyzerProcessor(config, None, config.get("db_config"))
    await processor.inicializar(None)

    event = Event(
        canal="settlement_data",
        datos={"data": zstd.compress(json.dumps({"roi_percent": 6.91, "total_trades": 10, "profits": [50, -20, 30]}).encode())},
        destino="system_analyzer"
    )
    await processor.manejar_evento(event)

    assert processor.metrics_cache["settlement_data"]["roi_percent"] == 6.91
    sharpe = await processor.calculate_sharpe_ratio([50, -20, 30])
    assert isinstance(sharpe, float)
    await processor.detener()