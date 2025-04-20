import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from corec.nucleus import CoreCNucleus


@pytest.fixture
def mock_redis():
    """Crea un cliente Redis simulado."""
    redis = AsyncMock()
    redis.xadd = AsyncMock()
    redis.close = AsyncMock()
    return redis


@pytest.fixture
def test_config():
    """Configuración de prueba para CoreCNucleus."""
    return {
        "instance_id": "test_corec",
        "db_config": {
            "dbname": "test_db",
            "user": "test_user",
            "password": "test_password",
            "host": "localhost",
            "port": 5432
        },
        "redis_config": {
            "host": "localhost",
            "port": 6379,
            "username": "test_user",
            "password": "test_password"
        },
        "bloques": [
            {
                "id": "test_block",
                "canal": 1,
                "entidades": 1000,
                "max_size_mb": 1,
                "entidades_por_bloque": 1000,
                "autoreparacion": {"max_errores": 0.05, "min_fitness": 0.2}
            }
        ],
        "plugins": {
            "test_plugin": {
                "enabled": True,
                "path": "plugins/test_plugin/config.json",
                "bloque": {
                    "bloque_id": "test_plugin_block",
                    "canal": 4,
                    "entidades": 500,
                    "max_size_mb": 2,
                    "max_errores": 0.05,
                    "min_fitness": 0.5
                }
            }
        }
    }


@pytest.fixture
async def nucleus(mock_redis, test_config):
    """Crea una instancia de CoreCNucleus para pruebas."""
    nucleus = CoreCNucleus("test_config.json")
    nucleus.config = test_config
    nucleus.redis_client = mock_redis
    try:
        yield nucleus
    finally:
        await nucleus.detener()


@pytest.fixture
def mock_postgresql():
    """Simula una conexión PostgreSQL."""
    conn = MagicMock()
    cur = MagicMock()
    conn.cursor.return_value = cur
    yield conn
