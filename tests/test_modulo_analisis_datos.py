import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch
from corec.modules.analisis_datos import ModuloAnalisisDatos

@pytest.fixture
def mock_nucleus():
    """Mock de núcleo para pruebas."""
    nucleus = AsyncMock()
    nucleus.config = {
        "analisis_datos_config": {
            "correlation_threshold": 0.8,
            "n_estimators": 100,
            "max_samples": 3
        }
    }
    nucleus.publicar_alerta = AsyncMock()
    return nucleus

@pytest.mark.asyncio
async def test_modulo_analisis_datos_inicializar(mock_nucleus):
    """Prueba la inicialización de ModuloAnalisisDatos."""
    analisis = ModuloAnalisisDatos()
    with patch.object(analisis.logger, "info") as mock_logger:
        await analisis.inicializar(mock_nucleus, mock_nucleus.config["analisis_datos_config"])
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_analisis_datos_analizar(mock_nucleus):
    """Prueba el análisis de datos en ModuloAnalisisDatos."""
    analisis = ModuloAnalisisDatos()
    await analisis.inicializar(mock_nucleus, mock_nucleus.config["analisis_datos_config"])
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
async def test_modulo_analisis_datos_detener(mock_nucleus):
    """Prueba la detención de ModuloAnalisisDatos."""
    analisis = ModuloAnalisisDatos()
    with patch.object(analisis.logger, "info") as mock_logger:
        await analisis.inicializar(mock_nucleus)
        await analisis.detener()
        assert mock_logger.called_with_call("[AnálisisDatos] Módulo detenido")
