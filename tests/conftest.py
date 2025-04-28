import pytest
from unittest.mock import AsyncMock, MagicMock
from corec.nucleus import CoreCNucleus
import pandas as pd

@pytest.fixture
def mock_redis():
    redis_mock = AsyncMock()
    redis_mock.rpush = AsyncMock()
    redis_mock.close = AsyncMock()
    return redis_mock

@pytest.fixture
def mock_db_pool():
    db_pool_mock = AsyncMock()
    conn_mock = AsyncMock()
    conn_mock.execute = AsyncMock()
    db_pool_mock.acquire.return_value.__aenter__ = AsyncMock(return_value=conn_mock)
    db_pool_mock.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    db_pool_mock.close = AsyncMock()
    return db_pool_mock

@pytest.fixture
def test_config():
    return {
        "bloques": [
            {"id": "test_block", "canal": 1, "entidades": [], "umbral": 0.0}
        ],
        "ia_config": {
            "enabled": False,
            "model_path": "",
            "pretrained": False,
            "n_classes": 3,
            "timeout_seconds": 2.0,
            "batch_size": 64,
            "max_size_mb": 50
        },
        "redis_config": {
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "postgresql_config": {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_pass"
        },
        "analisis_datos_config": {
            "max_samples": 1000,
            "n_estimators": 100,
            "correlation_threshold": 0.8
        }
    }

@pytest.fixture
def nucleus(test_config, mock_redis, mock_db_pool):
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool):
        nucleus = CoreCNucleus("config/corec_config.json")
        return nucleus
