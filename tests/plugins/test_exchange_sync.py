#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/plugins/test_exchange_sync.py
"""
test_exchange_sync.py
Pruebas unitarias para el plugin exchange_sync.
"""

import pytest
import asyncio
import json
import zstandard as zstd
from src.plugins.exchange_sync.processors.exchange_processor import ExchangeProcessor
from src.utils.config import load_secrets

@pytest.mark.asyncio
async def test_exchange_processor(monkeypatch):
    async def mock_get(self, url, **kwargs):
        class MockResponse:
            async def json(self):
                if "ticker/price" in url:
                    return {"price": "35000.0"}
                elif "openOrders" in url:
                    return [{"orderId": "123", "symbol": "BTCUSDT", "type": "spot", "status": "open"}]
            status = 200
        return MockResponse()

    async def mock_xadd(self, stream, data):
        pass

    async def mock_connect(self):
        return True

    async def mock_save_price(self, exchange, symbol, market, price, timestamp):
        pass

    async def mock_save_order(self, exchange, order_id, symbol, market, status, timestamp):
        pass

    monkeypatch.setattr("aiohttp.ClientSession.get", mock_get)
    monkeypatch.setattr("redis.asyncio.Redis.xadd", mock_xadd)
    monkeypatch.setattr("redis.asyncio.Redis.setex", lambda self, key, ttl, value: None)
    monkeypatch.setattr("src.plugins.exchange_sync.utils.db.ExchangeDB.connect", mock_connect)
    monkeypatch.setattr("src.plugins.exchange_sync.utils.db.ExchangeDB.save_price", mock_save_price)
    monkeypatch.setattr("src.plugins.exchange_sync.utils.db.ExchangeDB.save_order", mock_save_order)

    config = load_secrets("configs/plugins/exchange_sync/exchange_sync.yaml")
    processor = ExchangeProcessor(config, None, config.get("db_config"))
    await processor.inicializar(None)

    # Simular una iteración para un exchange
    await processor.fetch_exchange_data(config["exchange_config"]["exchanges"][0])

    await processor.detener()