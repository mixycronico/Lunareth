#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/plugins/test_macro_sync.py
"""
test_macro_sync.py
Pruebas unitarias para el plugin macro_sync.
"""

import pytest
import asyncio
import json
import zstandard as zstd
from src.plugins.macro_sync.processors.macro_processor import MacroProcessor
from src.utils.config import load_secrets

@pytest.mark.asyncio
async def test_macro_processor(monkeypatch):
    async def mock_get(self, url, **kwargs):
        class MockResponse:
            async def json(self):
                if "Alpha Vantage" in url:
                    return {"Global Quote": {"05. price": "4500.0"}}
                elif "CoinMarketCap" in url:
                    return {"data": [{"symbol": "SOL"}, {"symbol": "ADA"}]}
                elif "CoinDesk" in url:
                    return {"sentiment": 0.7}
            status = 200
        return MockResponse()

    async def mock_xadd(self, stream, data):
        pass

    async def mock_setex(self, key, ttl, value):
        pass

    async def mock_connect(self):
        return True

    async def mock_save_macro_data(self, sp500_price, nasdaq_price, vix_price, gold_price, oil_price, altcoins_volume, news_sentiment, timestamp):
        pass

    monkeypatch.setattr("aiohttp.ClientSession.get", mock_get)
    monkeypatch.setattr("redis.asyncio.Redis.xadd", mock_xadd)
    monkeypatch.setattr("redis.asyncio.Redis.setex", mock_setex)
    monkeypatch.setattr("src.plugins.macro_sync.utils.db.MacroDB.connect", mock_connect)
    monkeypatch.setattr("src.plugins.macro_sync.utils.db.MacroDB.save_macro_data", mock_save_macro_data)

    config = load_secrets("configs/plugins/macro_sync/macro_sync.yaml")
    processor = MacroProcessor(config, None, config.get("db_config"))
    await processor.inicializar(None)

    # Simular una iteración de fetch_macro_data
    await processor.fetch_macro_data()

    # Verificar caché
    assert processor.macro_data_cache["alpha_vantage"].get("spy_price", 0) == 4500.0
    assert len(processor.macro_data_cache["coinmarketcap"].get("altcoins", [])) == 2
    assert processor.macro_data_cache["coindesk"].get("news_sentiment", 0) == 0.7

    await processor.detener()