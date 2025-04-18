#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/plugins/test_market_monitor.py
"""
test_market_monitor.py
Pruebas unitarias para el plugin market_monitor.
"""

import pytest
import asyncio
import json
import zstandard as zstd
from src.plugins.market_monitor.processors.monitor_processor import MonitorProcessor
from src.utils.config import load_secrets
from corec.entidad_base import Event

@pytest.mark.asyncio
async def test_monitor_processor(monkeypatch):
    async def mock_xadd(self, stream, data):
        pass

    async def mock_connect(self):
        return True

    async def mock_save_price(self, symbol, price, timestamp):
        pass

    monkeypatch.setattr("redis.asyncio.Redis.xadd", mock_xadd)
    monkeypatch.setattr("src.plugins.market_monitor.utils.db.MonitorDB.connect", mock_connect)
    monkeypatch.setattr("src.plugins.market_monitor.utils.db.MonitorDB.save_price", mock_save_price)

    config = load_secrets("configs/plugins/market_monitor/market_monitor.yaml")
    processor = MonitorProcessor(config, None, config.get("db_config"))
    await processor.inicializar(None)

    # Simular evento macro_data
    macro_event = Event(
        canal="macro_data",
        datos={"altcoins": ["SOL/USDT", "ADA/USDT"]},
        destino="market_monitor"
    )
    await processor.manejar_evento(macro_event)
    assert processor.altcoins == ["SOL/USDT", "ADA/USDT"]

    # Simular evento exchange_data
    price_data = {"symbol": "BTC/USDT", "price": 35000.0, "timestamp": 1234567890.0}
    exchange_event = Event(
        canal="exchange_data",
        datos={"data": zstd.compress(json.dumps(price_data).encode())},
        destino="market_monitor"
    )
    await processor.manejar_evento(exchange_event)
    assert processor.prices["BTC/USDT"]["price"] == 35000.0

    # Simular una iteración de monitoreo
    await processor.monitor_prices()

    await processor.detener()