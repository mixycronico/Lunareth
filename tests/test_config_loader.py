import pytest
from corec.config_loader import load_config_dict, ConfigSchema
from pathlib import Path
import json

def test_load_config_valid(tmp_path):
    """Prueba la carga de una configuración válida."""
    config_data = {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",
            "user": "postgres",
            "password": "password",
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
        "bloques": [
            {
                "id": "test_block",
                "canal": 1,
                "entidades": 100,
                "max_size_mb": 1,
                "entidades_por_bloque": 1000,
                "autoreparacion": {
                    "max_errores": 0.05,
                    "min_fitness": 0.2
                }
            }
        ],
        "plugins": {}
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data))
    
    result = load_config_dict(str(config_path))
    assert isinstance(result, dict)
    assert result["instance_id"] == "corec1"
    assert len(result["bloques"]) == 1
    assert result["bloques"][0]["id"] == "test_block"

def test_load_config_invalid(tmp_path):
    """Prueba la carga de una configuración inválida."""
    config_data = {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",
            # Faltan campos requeridos
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data))
    
    with pytest.raises(ValueError, match="Invalid config format"):
        load_config_dict(str(config_path))

def test_load_config_missing_field(tmp_path):
    """Prueba la carga de una configuración con campos faltantes."""
    config_data = {
        # Falta instance_id
        "db_config": {
            "dbname": "corec_db",
            "user": "postgres",
            "password": "password",
            "host": "localhost",
            "port": 5432
        },
        "redis_config": {
            "host": "localhost",
            "port": 6379,
            "username": "corec_user",
            "password": "secure_password"
        }
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data))
    
    with pytest.raises(ValueError, match="Invalid config format"):
        load_config_dict(str(config_path))

def test_load_config_duplicate_block_ids(tmp_path):
    """Prueba la carga de una configuración con IDs de bloques duplicados."""
    config_data = {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",
            "user": "postgres",
            "password": "password",
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
        "bloques": [
            {
                "id": "test_block",
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
                "id": "test_block",  # ID duplicado
                "canal": 2,
                "entidades": 100,
                "max_size_mb": 1,
                "entidades_por_bloque": 1000,
                "autoreparacion": {
                    "max_errores": 0.05,
                    "min_fitness": 0.2
                }
            }
        ],
        "plugins": {}
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data))
    
    with pytest.raises(ValueError, match="Invalid config format"):
        load_config_dict(str(config_path))
