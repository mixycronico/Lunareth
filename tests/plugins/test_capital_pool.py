#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/plugins/test_capital_pool.py
"""
test_capital_pool.py
Pruebas unitarias para el plugin capital_pool.
"""

import pytest
import asyncio
import json
import zstandard as zstd
from src.plugins.capital_pool.processors.capital_processor import CapitalProcessor
from src.utils.config import load_secrets
from corec.entidad_base import Event

@pytest.mark.asyncio
async def test_capital_processor(monkeypatch):
    async def mock_xadd(self, stream, data):
        pass

    async def mock_connect(self):
        return True

    async def mock_save_contribution(self, user_id, amount, timestamp):
        pass

    async def mock_save_withdrawal(self, user_id, amount, timestamp):
        pass

    async def mock_update_pool(self, total):
        pass

    async def mock_update_active_capital(self, active):
        pass

    async def mock_get_pool_total(self):
        return 1000.0

    async def mock_get_active_capital(self):
        return 0.0

    async def mock_get_users(self):
        return {"user1": 500.0, "user2": 500.0}

    monkeypatch.setattr("redis.asyncio.Redis.xadd", mock_xadd)
    monkeypatch.setattr("src.plugins.capital_pool.utils.db.CapitalDB.connect", mock_connect)
    monkeypatch.setattr("src.plugins.capital_pool.utils.db.CapitalDB.save_contribution", mock_save_contribution)
    monkeypatch.setattr("src.plugins.capital_pool.utils.db.CapitalDB.save_withdrawal", mock_save_withdrawal)
    monkeypatch.setattr("src.plugins.capital_pool.utils.db.CapitalDB.update_pool", mock_update_pool)
    monkeypatch.setattr("src.plugins.capital_pool.utils.db.CapitalDB.update_active_capital", mock_update_active_capital)
    monkeypatch.setattr("src.plugins.capital_pool.utils.db.CapitalDB.get_pool_total", mock_get_pool_total)
    monkeypatch.setattr("src.plugins.capital_pool.utils.db.CapitalDB.get_active_capital", mock_get_active_capital)
    monkeypatch.setattr("src.plugins.capital_pool.utils.db.CapitalDB.get_users", mock_get_users)

    config = load_secrets("configs/plugins/capital_pool/capital_pool.yaml")
    processor = CapitalProcessor(config, None, config.get("db_config"))
    await processor.inicializar(None)

    # Simular contribución
    assert await processor.add_contribution("user3", 200.0)
    assert processor.pool == 1200.0

    # Simular retiro
    assert await processor.process_withdrawal("user1", 100.0)
    assert processor.pool == 1100.0

    # Simular resultado de trading
    result_event = Event(
        canal="trading_results",
        datos={"data": zstd.compress(json.dumps({"profit": 50.0, "quantity": 0.001, "price": 35000.0}).encode())},
        destino="capital_pool"
    )
    await processor.manejar_evento(result_event)
    assert processor.pool == 1150.0

    # Simular asignación de capital
    await processor.assign_capital()
    assert processor.active_capital > 0

    await processor.detener()