import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from corec.nucleus import CoreCNucleus
from corec.config_loader import load_config_dict
from pathlib import Path
import torch
import torch.nn as nn

@pytest.fixture
def test_config():
    """Configuración de prueba completa basada en corec_config.json."""
    return {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",  # Cambiado de database a dbname
            "user": "postgres",
            "password": "your_password",
            "host": "localhost",
            "port": 5432
        },
        "redis_config": {
            "host": "localhost",
            "port": 6379,
            "username": "corec_user",
            "password": "secure_password",
            "max_connections": 100,
            "stream_max_length": 5000
        },
        "ia_config": {
            "enabled": False,  # Disable IA to avoid model loading issues
            "model_path": "",  # Valid string for CoreCConfig
            "max_size_mb": 50,
            "pretrained": False,
            "n_classes": 3,
            "timeout_seconds": 5.0,
            "batch_size": 64
        },
        "analisis_datos_config": {
            "correlation_threshold": 0.8,
            "n_estimators": 100,
            "max_samples": 1000
        },
        "bloques": [
            {
                "id": "enjambre_sensor",
                "canal": 1,
                "entidades": 100,
                "max_size_mb": 1,
                "entidades_por_bloque": 1000,
                "autoreparacion": {
                    "max_errores": 0.05,
                    "min_fitness": 0.2
                }
            },
            {
                "id": "nodo_seguridad",
                "canal": 2,
                "entidades": 100,
                "max_size_mb": 1,
                "entidades_por_bloque": 1000,
                "autoreparacion": {
                    "max_errores": 0.02,
                    "min_fitness": 0.5
                }
            },
            {
                "id": "ia_analisis",
                "canal": 4,
                "entidades": 100,
                "max_size_mb": 50,
                "entidades_por_bloque": 100,
                "ia_timeout_seconds": 10.0,
                "autoreparacion": {
                    "max_errores": 0.1,
                    "min_fitness": 0.3
                }
            }
        ],
        "plugins": {
            "crypto_trading": {
                "enabled": True,
                "path": "plugins/crypto_trading/config.json",
                "bloque": {
                    "bloque_id": "trading_block",
                    "canal": 3,
                    "entidades": 2000,
                    "max_size_mb": 5,
                    "max_errores": 0.1,
                    "min_fitness": 0.3
                }
            },
            "test_plugin": {
                "enabled": True,
                "path": "plugins/test_plugin/config.json",
                "bloque": {
                    "bloque_id": "test_plugin",
                    "canal": 4,
                    "entidades": 500,
                    "max_size_mb": 5,
                    "max_errores": 0.1,
                    "min_fitness": 0.3
                }
            }
        }
    }

@pytest.fixture
def mock_redis():
    """Mock de cliente Redis con soporte para simular fallos."""
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = None
    redis.ping.return_value = True
    redis.xadd.return_value = None
    redis.xlen.return_value = 0
    redis.close.return_value = None
    redis.xread.return_value = []
    yield redis

@pytest.fixture
def mock_db_pool():
    """Mock de pool de conexiones PostgreSQL con soporte para simular fallos."""
    conn = AsyncMock()
    conn.execute.return_value = None
    conn.fetch.return_value = []
    conn.fetchrow.return_value = None
    conn.close.return_value = None
    pool = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    pool.acquire.return_value.__aexit__.return_value = None
    yield pool

@pytest.fixture
def mock_postgresql():
    """Mock de conexión síncrona compatible con psycopg2."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value = None
    cursor.close.return_value = None
    conn.cursor.return_value = cursor
    conn.commit.return_value = None
    conn.close.return_value = None
    yield conn

@pytest.fixture
async def nucleus(mock_redis, mock_db_pool, test_config, tmp_path):
    """Fixture para inicializar CoreCNucleus con mocks."""
    config_path = tmp_path / "test_config.json"
    config_path.write_text(json.dumps(test_config))

    try:
        with patch("corec.config_loader.load_config_dict", return_value=test_config), \
             patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
             patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
             patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()) as mock_schedule, \
             patch("pandas.DataFrame", MagicMock()) as mock_df, \
             patch("corec.utils.torch_utils.load_mobilenet_v3_small", return_value=MagicMock(spec=nn.Module)):
            mock_schedule.return_value = None
            nucleus = CoreCNucleus(str(config_path))
            await nucleus.inicializar()
            yield nucleus
            await nucleus.detener()
    except Exception as e:
        pytest.fail(f"Failed to initialize nucleus fixture: {e}")

@pytest.fixture
def mock_config():
    """Configuración mínima para pruebas."""
    return {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",  # Cambiado de database a dbname
            "user": "postgres",
            "password": "your_password",
            "host": "localhost",
            "port": 5432
        },
        "redis_config": {
            "host": "localhost",
            "port": 6379,
            "username": "corec_user",
            "password": "secure_password"
        },
        "ia_config": {
            "enabled": False,
            "model_path": "",
            "max_size_mb": 50,
            "pretrained": False,
            "n_classes": 3,
            "timeout_seconds": 5.0,
            "batch_size": 64
        },
        "analisis_datos_config": {
            "correlation_threshold": 0.8,
            "n_estimators": 100,
            "max_samples": 1000
        },
        "bloques": [],
        "plugins": {}
    }
