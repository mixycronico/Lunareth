import pytest
import time
from corec.modules.ia import ModuloIA
from corec.modules.analisis_datos import ModuloAnalisisDatos
from corec.blocks import BloqueSimbiotico
from corec.nucleus import CoreCNucleus
from unittest.mock import AsyncMock, patch
from plugins import PluginCommand
import pandas as pd

@pytest.mark.asyncio
async def test_integration_process_and_audit(nucleus):
    """Prueba la integración de procesamiento y auditoría."""
    with patch("corec.modules.ejecucion.ModuloEjecucion.encolar_bloque", AsyncMock()) as mock_encolar, \
         patch("corec.modules.auditoria.ModuloAuditoria.detectar_anomalias", AsyncMock()) as mock_detectar, \
         patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()):
        await nucleus.inicializar()
        await nucleus.process_bloque(nucleus.bloques[0])
        await nucleus.modules["auditoria"].detectar_anomalias()
        assert mock_encolar.called
        assert mock_detectar.called

@pytest.mark.asyncio
async def test_integration_synchronize_and_plugin_execution(nucleus):
    """Prueba la integración de sincronización y ejecución de plugins."""
    with patch("corec.modules.sincronizacion.ModuloSincronizacion.redirigir_entidades", AsyncMock()) as mock_synchronize:
        plugin_id = "crypto_trading"
        comando = {"action": "ejecutar_operacion", "params": {"exchange": "binance", "pair": "BTC/USDT", "side": "buy"}}
        plugin_mock = AsyncMock()
        plugin_mock.manejar_comando.return_value = {"status": "success"}
        nucleus.plugins[plugin_id] = plugin_mock
        await nucleus.inicializar()
        if len(nucleus.bloques) >= 2:
            await nucleus.modules["sincronizacion"].redirigir_entidades(
                nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal
            )
        assert mock_synchronize.called
        result = await nucleus.ejecutar_plugin(plugin_id, comando)
        assert result["status"] == "success"
        plugin_mock.manejar_comando.assert_called_once_with(comando)

@pytest.mark.asyncio
async def test_integration_ia_processing(nucleus, test_config):
    """Prueba la integración de procesamiento de IA."""
    ia_module = ModuloIA()
    await ia_module.inicializar(nucleus, test_config["ia_config"])
    bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus)
    datos = {"valores": [0.1, 0.2, 0.3]}
    with patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        result = await ia_module.procesar_bloque(bloque, datos)
        assert "mensajes" in result
        assert mock_alerta.called

@pytest.mark.asyncio
async def test_integration_analisis_datos(nucleus, test_config):
    """Prueba la integración de análisis de datos."""
    analisis = ModuloAnalisisDatos()
    config = test_config["analisis_datos_config"]
    await analisis.inicializar(nucleus, config)
    datos = pd.DataFrame({
        "valores": [0.1, 0.2, 0.3],
        "valores2": [0.4, 0.5, 0.6]
    })
    with patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        result = await analisis.analizar(datos, "test_datos")
        assert "estadisticas" in result
        assert mock_alerta.called

@pytest.mark.asyncio
async def test_integration_alert_archiving(nucleus, mock_redis, mock_db_pool):
    """Prueba la integración de archivado de alertas cuando el flujo de Redis está lleno."""
    mock_redis.xlen.return_value = 4500
    alerta = {
        "tipo": "test_alerta",
        "bloque_id": "test_block",
        "mensaje": "Test",
        "timestamp": time.time()
    }
    with patch.object(nucleus, "archive_alert", AsyncMock()) as mock_archive:
        await nucleus.publicar_alerta(alerta)
        assert mock_archive.called
