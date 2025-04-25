# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from corec.nucleus import CoreCNucleus

@pytest.fixture
def test_config():
    """Configuración de prueba completa basada en corec_config.json."""
    return {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",
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
            "enabled": True,
            "model_path": "corec/models/mobilev3/model.pth",
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
                    "bloque_id": "test_block",
                    "canal": 4,
                    "entidades": 500,
                    "max_size_mb": 1,
                    "max_errores": 0.05,
                    "min_fitness": 0.5
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
    yield redis

@pytest.fixture
def mock_db_pool():
    """Mock de pool de conexiones PostgreSQL con soporte para simular fallos."""
    conn = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value = None
    cursor.close.return_value = None
    conn.cursor.return_value = cursor
    conn.commit.return_value = None
    conn.close.return_value = None
    yield conn

@pytest.fixture
async def nucleus(mock_redis, mock_db_pool, test_config):
    """Fixture para inicializar CoreCNucleus con mocks."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.scheduler.Scheduler.schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        mock_schedule.return_value = None
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        yield nucleus
        await nucleus.detener()
