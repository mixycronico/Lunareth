import pytest
from corec.config_loader import load_config_dict, CoreCConfig

def test_load_config_valid(test_config):
    config = load_config_dict("config/corec_config.json")
    assert config["instance_id"] == "corec1"
    assert len(config["bloques"]) == 2
    assert config["plugins"]["crypto_trading"]["enabled"] is True
    assert config["redis_config"]["username"] == "corec_user"

def test_load_config_invalid():
    invalid_config = {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",
            "user": "postgres",
            "password": "your_password",
            "host": "localhost",
            "port": "invalid_port"  # Port debe ser un entero
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
    with pytest.raises(ValueError):
        CoreCConfig(**invalid_config)

def test_load_config_missing_field():
    invalid_config = {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",
            "user": "postgres",
            "password": "your_password",
            "host": "localhost",
            # Falta el campo "port"
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
    with pytest.raises(ValueError):
        CoreCConfig(**invalid_config)
