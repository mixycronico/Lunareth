import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from corec.nucleus import CoreCNucleus
import pandas as pd

@pytest.fixture
def mock_redis():
    redis_mock = AsyncMock()
    redis_mock.rpush = AsyncMock()
    redis_mock.close = AsyncMock()
    redis_mock.xadd = AsyncMock()
    redis_mock.xlen = AsyncMock(return_value=100)
    redis_mock.xread = AsyncMock(return_value=[])
    return redis_mock

@pytest.fixture
def mock_db_pool():
    db_pool_mock = AsyncMock()
    conn_mock = AsyncMock()
    conn_mock.execute = AsyncMock()
    conn_mock.fetch = AsyncMock(return_value=[])
    db_pool_mock.acquire.return_value.__aenter__ = AsyncMock(return_value=conn_mock)
    db_pool_mock.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    db_pool_mock.close = AsyncMock()
    return db_pool_mock

@pytest.fixture
def test_config():
    return {
        "bloques": [
            {
                "id": "test_block",
                "canal": 1,
                "entidades": 1,
                "max_size_mb": 50.0,
                "ia_timeout_seconds": 2.0
            },
            {
                "id": "ia_analisis",
                "canal": 4,
                "entidades": 1,
                "max_size_mb": 50.0,
                "ia_timeout_seconds": 0.1
            }
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
            "db": 0,
            "stream_max_length": 5000
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
        },
        "registro_config": {},
        "sincronizacion_config": {},
        "ejecucion_config": {},
        "auditoria_config": {}
    }

@pytest.fixture
def nucleus(test_config, mock_redis, mock_db_pool):
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool):
        nucleus = CoreCNucleus("config/corec_config.json")
        return nucleus
