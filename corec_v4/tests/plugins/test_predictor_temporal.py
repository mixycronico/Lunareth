#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/plugins/test_predictor_temporal.py
"""
test_predictor_temporal.py
Pruebas unitarias para el plugin predictor_temporal.
"""

import pytest
import asyncio
import json
from src.plugins.predictor_temporal.processors.predictor_processor import PredictorProcessor
from src.utils.config import load_secrets
from corec.entidad_base import Event

@pytest.mark.asyncio
async def test_predictor_processor(monkeypatch):
    async def mock_razonar(self, datos, contexto):
        return {"estado": "ok", "respuesta": "Análisis local"}

    async def mock_xadd(self, stream, data):
        pass

    async def mock_connect(self):
        return True

    async def mock_save_prediction(self, nano_id, symbol, prediction, actual_value, error, macro_context, timestamp):
        pass

    async def mock_save_metrics(self, nano_id, mse, mae, predictions_count, timestamp):
        pass

    monkeypatch.setattr("src.core.nucleus.CoreCNucleus.razonar", mock_razonar)
    monkeypatch.setattr("redis.asyncio.Redis.xadd", mock_xadd)
    monkeypatch.setattr("src.plugins.predictor_temporal.utils.db.PredictorDB.connect", mock_connect)
    monkeypatch.setattr("src.plugins.predictor_temporal.utils.db.PredictorDB.save_prediction", mock_save_prediction)
    monkeypatch.setattr("src.plugins.predictor_temporal.utils.db.PredictorDB.save_metrics", mock_save_metrics)

    config = load_secrets("configs/plugins/predictor_temporal/predictor_temporal.yaml")
    processor = PredictorProcessor(config, None, config.get("db_config"))
    await processor.inicializar(None)

    # Simular evento macro_data
    macro_event = Event(
        canal="macro_data",
        datos={"sp500_price": 4500.0},
        destino="predictor_temporal"
    )
    await processor.manejar_evento(macro_event)

    # Procesar datos
    result = await processor.procesar(
        {"valores": [35000.0] * 60, "symbol": "BTC/USDT", "actual_value": 35100.0},
        {"timestamp": 1234567890, "canal": "predictor_temporal", "nano_id": "test", "instance_id": "corec1"}
    )

    assert result["estado"] == "ok"
    assert "prediction" in result
    assert result["symbol"] == "BTC/USDT"
    assert result["analisis"] == "Análisis local"
    assert result["error"] is not None

    await processor.detener()