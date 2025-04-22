import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from corec.nucleus import CoreCNucleus
from corec.entities import Entidad

# Clase EntidadConError para simular un error al cambiar el estado
class EntidadConError(Entidad):
    def __init__(self, id: str, canal: int, procesar_func):
        self._estado = "inactiva"
        self.id = id
        self.canal = canal
        self.procesar_func = procesar_func

    @property
    def estado(self):
        return self._estado

    @estado.setter
    def estado(self, value):
        if value == "activa":
            raise Exception("Error al asignar estado")
        self._estado = value

@pytest.fixture
def mock_redis():
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = None
    redis.ping.return_value = True
    redis.xadd.return_value = None
    redis.close.return_value = None
    yield redis

@pytest.fixture
def mock_db_pool():
    db_pool = AsyncMock()
    conn = AsyncMock()
    # Simulamos el método execute para que sea un coroutine que devuelve None
    async def mock_execute(*args, **kwargs):
        return None
    conn.execute = AsyncMock(side_effect=mock_execute)
    db_pool.acquire.return_value.__aenter__.return_value = conn
    db_pool.close.return_value = None
    yield db_pool

@pytest.fixture
def mock_postgresql():
    db_pool = AsyncMock()
    conn = AsyncMock()
    # Simulamos el método execute para que sea un coroutine que devuelve None
    async def mock_execute(*args, **kwargs):
        return None
    conn.execute = AsyncMock(side_effect=mock_execute)
    db_pool.acquire.return_value.__aenter__.return_value = conn
    db_pool.close.return_value = None
    yield db_pool

@pytest.fixture
def test_config():
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
            "password": "secure_password"
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
                "bloque": {"id": "test_plugin", "canal": 4, "entidades": 500}
            }
        }
    }

@pytest.fixture
async def nucleus(mock_redis, mock_db_pool, test_config):
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.nucleus.init_postgresql", return_value=mock_db_pool), \
         patch("aioredis.from_url", return_value=mock_redis):
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        yield nucleus
        await nucleus.detener()

@pytest.fixture
def mock_config():
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
            "password": "secure_password"
        },
        "bloques": [],
        "plugins": {}
    }
