import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from plugins.crypto_trading.processors.orchestrator_processor import OrchestratorProcessor
from plugins.crypto_trading.utils.db import TradingDB

@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = None
    redis.ping.return_value = True
    yield redis

@pytest.fixture
def mock_trading_db():
    db = AsyncMock()
    db.pool = AsyncMock()
    db.pool.acquire.return_value.__aenter__.return_value = AsyncMock()
    yield db

@pytest.fixture
def mock_nucleus():
    nucleus = MagicMock()
    nucleus.redis_client = AsyncMock()
    nucleus.publicar_alerta = AsyncMock()
    nucleus.bloques = []
    yield nucleus

@pytest.fixture
def config():
    return {
        "total_capital": 1000,
        "paper_mode": True,
        "exchanges": [
            {"name": "binance", "api_key": "test_key", "api_secret": "test_secret"},
            {"name": "bybit", "api_key": "test_key", "api_secret": "test_secret"}
        ],
        "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
        "db_config": {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_pass",
            "host": "localhost",
            "port": 5432
        },
        "cb_max_failures": 3,
        "cb_reset_timeout": 900
    }

@pytest.fixture
async def orchestrator(mock_nucleus, mock_redis, mock_trading_db, config):
    orchestrator = OrchestratorProcessor(config, mock_nucleus, mock_redis)
    await orchestrator.initialize()
    yield orchestrator
    await orchestrator.detener()
