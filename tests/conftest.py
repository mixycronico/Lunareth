import pytest
from unittest.mock import AsyncMock, MagicMock
from corec.nucleus import CoreCNucleus


test_config = {
    "db_config": {
        "dbname": "test_db",
        "user": "test_user",
        "password": "test_pass",
        "host": "localhost",
        "port": 5432
    },
    "redis_config": {
        "host": "localhost",
        "port": 6379
    },
    "bloques": [
        {"id": "test_block", "canal": 1, "entidades": 1000}
    ],
    "plugins": {
        "test_plugin": {
            "enabled": True,
            "bloque": {"id": "test_plugin", "canal": 4, "entidades": 500}
        }
    }
}


@pytest.fixture
def nucleus():
    nucleus = CoreCNucleus("test_config.yaml")
    nucleus.config = test_config
    yield nucleus


@pytest.fixture
def mock_redis():
    return AsyncMock()


@pytest.fixture
def mock_postgresql():
    mock = MagicMock()
    mock.cursor.return_value = MagicMock()
    return mock
