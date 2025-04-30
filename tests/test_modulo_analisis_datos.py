import pytest
import pandas as pd
from unittest.mock import AsyncMock, patch
from corec.modules.analisis_datos import ModuloAnalisisDatos
from corec.config_loader import CoreCConfig

@pytest.fixture
def mock_nucleus():
    """Mock de núcleo para pruebas."""
    nucleus = AsyncMock()
    nucleus.config = CoreCConfig(
        instance_id="corec1",
        db_config={"dbname": "corec_db", "user": "postgres", "password": "password", "host": "localhost", "port": 5432},
        redis_config={"host": "localhost", "port": 6379, "username": "corec_user", "password": "secure_password"},
        ia_config={"enabled": False, "model_path": "", "max_size_mb": 50, "pretrained": False, "n_classes": 3, "timeout_seconds": 5.0, "batch_size": 64},
        analisis_datos_config={"correlation_threshold": 0.8, "n_estimators": 100, "max_samples": 3},
        ml_config={"enabled": True, "model_type": "linear_regression", "historial_size": 50, "min_samples_train": 10},
        autosanacion_config={"enabled": True, "check_interval_seconds": 120, "max_retries": 5, "retry_delay_min": 2, "retry_delay_max": 10},
        cognitivo_config={},
        bloques=[],
        plugins={}
    )
    nucleus.publicar_alerta = AsyncMock()
    return nucleus

@pytest.mark.asyncio
async def test_modulo_analisis_datos_inicializar(mock_nucleus):
    """Prueba la inicialización de ModuloAnalisisDatos."""
    analisis = ModuloAnalisisDatos()
    with patch.object(analisis.logger, "info") as mock_logger:
        await analisis.inicializar(mock_nucleus, mock_nucleus.config.analisis_datos_config)
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_analisis_datos_analizar(mock_nucleus):
    """Prueba el análisis de datos en ModuloAnalisisDatos."""
    analisis = ModuloAnalisisDatos()
    await analisis.inicializar(mock_nucleus, mock_nucleus.config.analisis_datos_config)
    df = pd.DataFrame({
        "enjambre_sensor": [0.5, 0.6, 0.7],
        "nodo_seguridad": [0.7, 0.8, 0.9]
    })
    with patch.object(mock_nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        result = await analisis.analizar(df, "test_analisis")
        assert "estadisticas" in result
        assert "num_anomalias" in result["anomalias"]
        assert mock_alerta.called

@pytest.mark.asyncio
async def test_modulo_analisis_datos_analizar_datos_vacios(mock_nucleus):
    """Prueba el análisis de datos vacíos en ModuloAnalisisDatos."""
    analisis = ModuloAnalisisDatos()
    await analisis.inicializar(mock_nucleus, mock_nucleus.config.analisis_datos_config)
    df = pd.DataFrame({})
    with patch.object(mock_nucleus, "publicar_alerta", AsyncMock()) as mock_alerta, \
         patch.object(analisis.logger, "warning") as mock_logger:
        result = await analisis.analizar(df, "test_analisis")
        assert result["estadisticas"] == {}
        assert result["correlaciones"] == {}
        assert result["anomalias"]["num_anomalias"] == 0
        assert mock_logger.called
        assert mock_alerta.called

@pytest.mark.asyncio
async def test_modulo_analisis_datos_analizar_error(mock_nucleus):
    """Prueba el manejo de errores durante el análisis en ModuloAnalisisDatos."""
    analisis = ModuloAnalisisDatos()
    await analisis.inicializar(mock_nucleus, mock_nucleus.config.analisis_datos_config)
    df = pd.DataFrame({
        "enjambre_sensor": [0.5, 0.6, 0.7],
        "nodo_seguridad": [0.7, 0.8, 0.9]
    })
    with patch("pandas.DataFrame.corr", side_effect=Exception("Error de correlación")) as mock_corr, \
         patch.object(mock_nucleus, "publicar_alerta", AsyncMock()) as mock_alerta, \
         patch.object(analisis.logger, "error") as mock_logger:
        result = await analisis.analizar(df, "test_analisis")
        assert result["correlaciones"] == {}
        assert mock_logger.called
        assert mock_alerta.called
        assert mock_alerta.call_args[0][0]["tipo"] == "error_analisis_datos"

@pytest.mark.asyncio
async def test_modulo_analisis_datos_detener(mock_nucleus):
    """Prueba la detención de ModuloAnalisisDatos."""
    analisis = ModuloAnalisisDatos()
    with patch.object(analisis.logger, "info") as mock_logger:
        await analisis.inicializar(mock_nucleus)
        await analisis.detener()
        assert mock_logger.called_with_call("Módulo AnalisisDatos detenido")
