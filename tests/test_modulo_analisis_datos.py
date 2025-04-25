# tests/test_modulo_analisis_datos.py
import pytest
import asyncio
import pandas as pd
from unittest.mock import AsyncMock, patch
from corec.modules.analisis_datos import ModuloAnalisisDatos

@pytest.mark.asyncio
async def test_modulo_analisis_datos_inicializar(nucleus):
    """Prueba la inicialización de ModuloAnalisisDatos."""
    analisis = ModuloAnalisisDatos()
    with patch.object(analisis.logger, "info") as mock_logger:
        await analisis.inicializar(nucleus, nucleus.config["analisis_datos_config"])
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_analisis_datos_analizar(nucleus):
    """Prueba el análisis de datos en ModuloAnalisisDatos."""
    analisis = ModuloAnalisisDatos()
    await analisis.inicializar(nucleus, nucleus.config["analisis_datos_config"])
    df = pd.DataFrame({
        "enjambre_sensor": [0.5, 0.6, 0.7],
        "nodo_seguridad": [0.7, 0.8, 0.9]
    })
    with patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        result = await analisis.analizar(df, "test_analisis")
        assert "estadisticas" in result
        assert "num_anomalías" in result
        assert "correlaciones" in result
        assert mock_alerta.call_count == 3  # Estadísticas, anomalías, correlaciones
