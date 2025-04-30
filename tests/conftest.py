import pytest
import json
from pathlib import Path
from corec.config_loader import load_config, CoreCConfig

@pytest.fixture
def valid_config_data():
    return {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",
            "user": "postgres",
            "password": "password",
            "host": "localhost",
            "port": 5432,
            "max_connections": 50,
            "connection_timeout": 10
        },
        "redis_config": {
            "host": "localhost",
            "port": 6379,
            "username": "corec_user",
            "password": "secure_password",
            "max_connections": 100,
            "stream_max_length": 5000,
            "retry_interval": 2
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
        "ml_config": {
            "enabled": True,
            "model_type": "linear_regression",
            "historial_size": 50,
            "min_samples_train": 10
        },
        "autosanacion_config": {
            "enabled": True,
            "check_interval_seconds": 120,
            "max_retries": 5,
            "retry_delay_min": 2,
            "retry_delay_max": 10
        },
        "cognitivo_config": {
            "max_memoria": 1000,
            "umbral_confianza": 0.5,
            "penalizacion_intuicion": 0.9,
            "max_percepciones": 5000,
            "impacto_adaptacion": 0.1,
            "confiabilidad_minima": 0.4,
            "umbral_afectivo_positivo": 0.8,
            "umbral_afectivo_negativo": -0.8,
            "peso_afectivo": 0.2,
            "umbral_fallo": 0.3,
            "peso_semantico": 0.1,
            "umbral_cambio_significativo": 0.05,
            "tasa_aprendizaje_minima": 0.1,
            "umbral_relevancia": 0.3,
            "peso_novedad": 0.3
        },
        "bloques": [
            {
                "id": "test_block",
                "canal": 1,
                "entidades": 100,
                "max_size_mb": 1,
                "entidades_por_bloque": 1000,
                "quantization_step": 0.1,
                "max_concurrent_tasks": 100,
                "cpu_intensive": False,
                "autoreparacion": {
                    "max_errores": 0.05,
                    "min_fitness": 0.2
                },
                "mutacion": {
                    "enabled": True,
                    "min_fitness": 0.2,
                    "mutation_rate": 0.1
                },
                "autorreplicacion": {
                    "enabled": True,
                    "max_entidades": 5000,
                    "min_fitness_trigger": 0.1
                }
            }
        ],
        "plugins": {}
    }

def test_load_config_valid(tmp_path, valid_config_data):
    """Prueba la carga de una configuración válida."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(valid_config_data))
    
    result = load_config(str(config_path))
    assert isinstance(result, CoreCConfig)
    assert result.instance_id == "corec1"
    assert len(result.bloques) == 1
    assert result.bloques[0].id == "test_block"

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
        load_config(str(config_path))

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
        load_config(str(config_path))

def test_load_config_duplicate_block_ids(tmp_path):
    """Prueba la carga de una configuración con IDs de bloques duplicados."""
    config_data = {
        "instance_id": "corec1",
        "db_config": {
            "dbname": "corec_db",
            "user": "postgres",
            "password": "password",
            "host": "localhost",
            "port": 5432,
            "max_connections": 50,
            "connection_timeout": 10
        },
        "redis_config": {
            "host": "localhost",
            "port": 6379,
            "username": "corec_user",
            "password": "secure_password",
            "max_connections": 100,
            "stream_max_length": 5000,
            "retry_interval": 2
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
        "ml_config": {
            "enabled": True,
            "model_type": "linear_regression",
            "historial_size": 50,
            "min_samples_train": 10
        },
        "autosanacion_config": {
            "enabled": True,
            "check_interval_seconds": 120,
            "max_retries": 5,
            "retry_delay_min": 2,
            "retry_delay_max": 10
        },
        "cognitivo_config": {
            "max_memoria": 1000,
            "umbral_confianza": 0.5,
            "penalizacion_intuicion": 0.9,
            "max_percepciones": 5000,
            "impacto_adaptacion": 0.1,
            "confiabilidad_minima": 0.4,
            "umbral_afectivo_positivo": 0.8,
            "umbral_afectivo_negativo": -0.8,
            "peso_afectivo": 0.2,
            "umbral_fallo": 0.3,
            "peso_semantico": 0.1,
            "umbral_cambio_significativo": 0.05,
            "tasa_aprendizaje_minima": 0.1,
            "umbral_relevancia": 0.3,
            "peso_novedad": 0.3
        },
        "bloques": [
            {
                "id": "test_block",
                "canal": 1,
                "entidades": 100,
                "max_size_mb": 1,
                "entidades_por_bloque": 1000,
                "quantization_step": 0.1,
                "max_concurrent_tasks": 100,
                "cpu_intensive": False,
                "autoreparacion": {
                    "max_errores": 0.05,
                    "min_fitness": 0.2
                },
                "mutacion": {
                    "enabled": True,
                    "min_fitness": 0.2,
                    "mutation_rate": 0.1
                },
                "autorreplicacion": {
                    "enabled": True,
                    "max_entidades": 5000,
                    "min_fitness_trigger": 0.1
                }
            },
            {
                "id": "test_block",  # ID duplicado
                "canal": 2,
                "entidades": 100,
                "max_size_mb": 1,
                "entidades_por_bloque": 1000,
                "quantization_step": 0.1,
                "max_concurrent_tasks": 100,
                "cpu_intensive": False,
                "autoreparacion": {
                    "max_errores": 0.05,
                    "min_fitness": 0.2
                },
                "mutacion": {
                    "enabled": True,
                    "min_fitness": 0.2,
                    "mutation_rate": 0.1
                },
                "autorreplicacion": {
                    "enabled": True,
                    "max_entidades": 5000,
                    "min_fitness_trigger": 0.1
                }
            }
        ],
        "plugins": {}
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data))
    
    with pytest.raises(ValueError, match="Duplicate block IDs found in configuration"):
        load_config(str(config_path))

def test_load_config_with_env_vars(tmp_path, valid_config_data, monkeypatch):
    """Prueba la carga de configuración con variables de entorno."""
    monkeypatch.setenv("DB_PASSWORD", "env_db_password")
    monkeypatch.setenv("REDIS_PASSWORD", "env_redis_password")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(valid_config_data))
    
    result = load_config(str(config_path))
    assert result.db_config.password == "env_db_password"
    assert result.redis_config.password == "env_redis_password"
