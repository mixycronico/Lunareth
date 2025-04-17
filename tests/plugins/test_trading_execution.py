#!/usr/bin/env python3
# tests/plugins/test_trading_execution.py
"""
test_trading_execution.py
Pruebas unitarias para el plugin trading_execution.
"""

import pytest
import asyncio
import json
import zstandard as zstd
from src.plugins.trading_execution.processors.execution_processor import ExecutionProcessor
from src.utils.config import load_secrets
from corec.entidad_base import Event

@pytest.mark.asyncio
async def test_execution_processor(monkeypatch):
    async def mock_connect(self):
        return True

    async def mock_xadd(self, stream, data):
        pass

    monkeypatch.setattr("src.plugins.trading_execution.utils.db.ExecutionDB.connect", mock_connect)
    monkeypatch.setattr("redis.asyncio.Redis.xadd", mock_xadd)

    config = load_secrets("configs/plugins/trading_execution/trading_execution.yaml")
    processor = ExecutionProcessor(config, None, config.get("db_config"))
    await processor.inicializar(None)

    prices = [1000, 1010, 1020, 990, 980, 970, 960, 950, 940, 930] * 100
    result = await processor.run_backtest({"symbol": "BTC/USDT", "risk": 0.02, "trades": 10})
    assert result["roi"] >= 0
    assert isinstance(result["sharpe_ratio"], float)

    await processor.detener()