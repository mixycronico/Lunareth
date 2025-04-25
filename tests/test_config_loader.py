# tests/test_config_loader.py
import pytest
from corec.config_loader import load_config_dict, CoreCConfig
from pydantic import ValidationError

def test_load_config_valid(test_config):
    """Prueba la carga de una configuración válida."""
    config = test_config
    assert config["instance_id"] == "corec1"
    assert len(config["bloques"]) == 3
    assert config["plugins"]["crypto_trading"]["enabled"] is True
    assert config["redis_config"]["username"] == "corec_user"
    assert config["bloques"][2]["ia_timeout_seconds"] == 10.0

def test_load_config_invalid():
    """Prueba la carga de una configuración con un puerto inválido."""
    invalid_config = {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",
            "user": "postgres",
            "password": "your_password",
            "host": "localhost",
            "port": "invalid_port"
        },
        "redis_config": {
            "host": "localhost",
            "port": 6379,
            "username": "corec_user",
            "password": "secure_password",
            "max_connections": 100,
            "stream_max_length": 5000
        },
        "bloques": [],
        "plugins": {}
    }
    with pytest.raises(ValidationError):
        CoreCConfig(**invalid_config)

def test_load_config_missing_field():
    """Prueba la carga de una configuración con un campo faltante."""
    invalid_config = {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",
            "user": "postgres",
            "password": "your_password",
            "host": "localhost"
        },
        "redis_config": {
            "host": "localhost",
            "port": 6379,
            "username": "corec_user",
            "password": "secure_password",
            "max_connections": 100,
            "stream_max_length": 5000
        },
        "bloques": [],
        "plugins": {}
    }
    with pytest.raises(ValidationError):
        CoreCConfig(**invalid_config)

def test_load_config_duplicate_block_ids():
    """Prueba la carga de una configuración con IDs de bloques duplicados."""
    invalid_config = {
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
        "bloques": [
            {
                "id": "enjambre_sensor",
                "canal": 1,
                "entidades": 100,
                "max_size_mb": 1,
                "entidades_por_bloque": 1000
            },
            {
                "id": "enjambre_sensor",
                "canal": 2,
                "entidades": 100,
                "max_size_mb": 1,
                "entidades_por_bloque": 1000
            }
        ],
        "plugins": {}
    }
    with pytest.raises(ValueError, match="IDs de bloques duplicados encontrados"):
        CoreCConfig(**invalid_config)
