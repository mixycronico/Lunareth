import pytest
import asyncio
import json
from pathlib import Path
import pandas as pd
from unittest.mock import AsyncMock, patch, MagicMock
from corec.nucleus import CoreCNucleus
from corec.blocks import BloqueSimbiotico
from corec.entities_superpuestas import EntidadSuperpuesta
from corec.config_loader import CoreCConfig

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
                "entidades": 1,
                "max_size_mb": 50.0,
                "entidades_por_bloque": 1000,
                "quantization_step": 0.1,
                "max_concurrent_tasks": 100,
                "cpu_intensive": False,
                "ia_timeout_seconds": 2.0,
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
                "id": "ia_analisis",
                "canal": 4,
                "entidades": 1,
                "max_size_mb": 50.0,
                "entidades_por_bloque": 100,
                "quantization_step": 0.01,
                "max_concurrent_tasks": 20,
                "cpu_intensive": True,
                "ia_timeout_seconds": 0.1,
                "autoreparacion": {
                    "max_errores": 0.1,
                    "min_fitness": 0.3
                },
                "mutacion": {
                    "enabled": True,
                    "min_fitness": 0.3,
                    "mutation_rate": 0.2
                },
                "autorreplicacion": {
                    "enabled": False,
                    "max_entidades": 100,
                    "min_fitness_trigger": 0.2
                }
            }
        ],
        "plugins": {},
        "quantization_step_default": 0.05,
        "max_enlaces_por_entidad": 100,
        "redis_stream_key": "corec:entrelazador",
        "alert_threshold": 0.9,
        "max_fallos_criticos": 0.5,
        "cpu_autoadjust_threshold": 0.9,
        "ram_autoadjust_threshold": 0.9,
        "concurrent_tasks_min": 10,
        "concurrent_tasks_max": 1000,
        "concurrent_tasks_reduction_factor": 0.8,
        "concurrent_tasks_increment_factor_default": 1.05,
        "cpu_stable_cycles": 3,
        "cpu_readings": 3,
        "cpu_reading_interval": 0.05,
        "performance_history_size": 10,
        "performance_threshold": 0.5,
        "increment_factor_min": 1.01,
        "increment_factor_max": 1.1
    }

@pytest.fixture
def nucleus(test_config, mock_redis, mock_db_pool):
    with patch("corec.config_loader.load_config", return_value=CoreCConfig(**test_config)), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool):
        nucleus = CoreCNucleus("config/corec_config.json")
        return nucleus

@pytest.mark.asyncio
async def test_nucleus_fallback_storage(test_config, mock_redis, mock_db_pool):
    """Prueba el almacenamiento en fallback cuando PostgreSQL falla."""
    test_config["ia_config"]["enabled"] = False
    with patch("corec.config_loader.load_config", return_value=CoreCConfig(**test_config)), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("apscheduler.schedulers.asyncio.AsyncIOScheduler.add_job", AsyncMock()), \
         patch("pandas.DataFrame", return_value=pd.DataFrame({"valores": [0.1, 0.2, 0.3]}, dtype=float)):
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        messages = [{
            "entidad_id": "ent_1",
            "canal": 1,
            "valor": 0.5,
            "clasificacion": "test",
            "probabilidad": 0.9,
            "timestamp": 12345,
            "roles": {}
        }]
        with patch("json.dump", MagicMock()) as mock_json_dump:
            await nucleus.save_fallback_messages(bloque_id="test_block", mensajes=messages)
            assert mock_json_dump.called
        await nucleus.detener()

@pytest.mark.asyncio
async def test_nucleus_retry_fallback(test_config, mock_redis, mock_db_pool, tmp_path):
    """Prueba el reintento de mensajes desde fallback a PostgreSQL."""
    test_config["ia_config"]["enabled"] = False
    with patch("corec.config_loader.load_config", return_value=CoreCConfig(**test_config)), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("apscheduler.schedulers.asyncio.AsyncIOScheduler.add_job", AsyncMock()), \
         patch("pandas.DataFrame", return_value=pd.DataFrame({"valores": [0.1, 0.2, 0.3]}, dtype=float)):
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        nucleus.db_pool = mock_db_pool
        fallback_file = tmp_path / "fallback_messages.json"
        messages = [{
            "bloque_id": "enjambre_sensor",
            "mensaje": {
                "entidad_id": "ent_1",
                "canal": 1,
                "valor": 0.5,
                "clasificacion": "test",
                "probabilidad": 0.9,
                "timestamp": 12345.0,
                "roles": {}
            },
            "retry_count": 0
        }]
        with open(fallback_file, "w") as f:
            json.dump(messages, f)
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value=None)
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        nucleus.fallback_storage = fallback_file
        nucleus.logger.info(f"[Debug] Attempting to retry fallback messages from {fallback_file}")
        await nucleus.retry_fallback_messages()
        nucleus.logger.info(f"[Debug] File exists after retry: {fallback_file.exists()}")
        assert not fallback_file.exists()
        assert conn.execute.called
        assert conn.execute.call_args[0][0].startswith("INSERT INTO mensajes")
        assert conn.execute.call_args[1][0] == "enjambre_sensor"
        assert conn.execute.call_args[1][1] == "ent_1"

@pytest.mark.asyncio
async def test_nucleus_global_concurrent_tasks_limit(test_config, mock_redis, mock_db_pool):
    """Prueba el límite global de tareas concurrentes."""
    test_config["ia_config"]["enabled"] = False
    with patch("corec.config_loader.load_config", return_value=CoreCConfig(**test_config)), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("apscheduler.schedulers.asyncio.AsyncIOScheduler.add_job", AsyncMock()), \
         patch("pandas.DataFrame", return_value=pd.DataFrame({"valores": [0.1, 0.2, 0.3]}, dtype=float)):
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        entidades = [
            EntidadSuperpuesta(
                f"ent_{i}",
                {"rol1": 0.5, "rol2": 0.5},
                quantization_step=0.05,
                min_fitness=0.3,
                mutation_rate=0.1,
                nucleus=nucleus
            ) for i in range(1)
        ]
        bloque = BloqueSimbiotico(
            id="test_block",
            canal=1,
            entidades=entidades,
            max_size_mb=10.0,
            nucleus=nucleus,
            quantization_step=0.05,
            max_errores=0.1,
            max_concurrent_tasks=1000
        )
        nucleus.global_concurrent_tasks = nucleus.global_concurrent_tasks_max
        with patch.object(nucleus.logger, "warning") as mock_logger, \
             patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
            await nucleus.process_bloque(bloque)
            assert mock_logger.called
            assert mock_alerta.called
            assert mock_alerta.call_args[0][0]["tipo"] == "limite_tareas_global"
        await nucleus.detener()

@pytest.mark.asyncio
async def test_nucleus_retry_fallback_max_retries(test_config, mock_redis, mock_db_pool, tmp_path):
    """Prueba el descarte de mensajes de fallback tras alcanzar el máximo de reintentos."""
    test_config["ia_config"]["enabled"] = False
    with patch("corec.config_loader.load_config", return_value=CoreCConfig(**test_config)), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("apscheduler.schedulers.asyncio.AsyncIOScheduler.add_job", AsyncMock()), \
         patch("pandas.DataFrame", return_value=pd.DataFrame({"valores": [0.1, 0.2, 0.3]}, dtype=float)):
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        nucleus.db_pool = mock_db_pool
        fallback_file = tmp_path / "fallback_messages.json"
        messages = [{
            "bloque_id": "enjambre_sensor",
            "mensaje": {
                "entidad_id": "ent_1",
                "canal": 1,
                "valor": 0.5,
                "clasificacion": "test",
                "probabilidad": 0.9,
                "timestamp": 12345.0,
                "roles": {}
            },
            "retry_count": 5  # Máximo de reintentos alcanzado
        }]
        with open(fallback_file, "w") as f:
            json.dump(messages, f)
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value=None)
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        nucleus.fallback_storage = fallback_file
        with patch.object(nucleus.logger, "warning") as mock_logger:
            await nucleus.retry_fallback_messages()
            assert not fallback_file.exists()
            assert not conn.execute.called
            assert mock_logger.called
            assert "Mensaje descartado tras 5 intentos" in str(mock_logger.call_args)
        await nucleus.detener()

@pytest.mark.asyncio
async def test_nucleus_reconexion_autosanacion(test_config, mock_redis, mock_db_pool):
    """Prueba la reconexión automática manejada por ModuloAutosanacion."""
    test_config["ia_config"]["enabled"] = False
    with patch("corec.config_loader.load_config", return_value=CoreCConfig(**test_config)), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("apscheduler.schedulers.asyncio.AsyncIOScheduler.add_job", AsyncMock()), \
         patch("pandas.DataFrame", return_value=pd.DataFrame({"valores": [0.1, 0.2, 0.3]}, dtype=float)):
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        nucleus.db_pool = None
        nucleus.redis_client = None
        with patch("corec.utils.db_utils.init_postgresql", AsyncMock(return_value=mock_db_pool)) as mock_init_postgresql, \
             patch("corec.utils.db_utils.init_redis", AsyncMock(return_value=mock_redis)) as mock_init_redis, \
             patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
            await nucleus.modules["autosanacion"].verificar_estado()
            assert mock_init_postgresql.called
            assert mock_init_redis.called
            assert mock_alerta.called
        await nucleus.detener()
