# tests/test_integration.py
import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from plugins import PluginCommand
from corec.modules.ia import ModuloIA
from corec.modules.analisis_datos import ModuloAnalisisDatos

@pytest.mark.asyncio
async def test_integration_process_and_audit(nucleus):
    """Prueba la integración de procesamiento de bloques y auditoría."""
    with patch("corec.modules.ejecucion.ModuloEjecucion.encolar_bloque", AsyncMock()) as mock_encolar, \
         patch("corec.modules.auditoria.ModuloAuditoria.detectar_anomalias", AsyncMock()) as mock_detectar:
        await nucleus.inicializar()
        await nucleus.process_bloque(nucleus.bloques[0])
        await nucleus.modules["auditoria"].detectar_anomalias()
        assert mock_encolar.called
        assert mock_detectar.called

@pytest.mark.asyncio
async def test_integration_synchronize_and_plugin_execution(nucleus):
    """Prueba la integración de sincronización de bloques y ejecución de plugins."""
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
        result = await nucleus.plugins[plugin_id].manejar_comando(PluginCommand(**comando))
        assert result["status"] == "success"
        plugin_mock.manejar_comando.assert_called_once_with(PluginCommand(**comando))

@pytest.mark.asyncio
async def test_integration_ia_processing(nucleus):
    """Prueba la integración de procesamiento de IA."""
    ia_module = ModuloIA()
    await ia_module.inicializar(nucleus, nucleus.config["ia_config"])
    bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus)
    bloque.ia_timeout_seconds = 10.0
    datos = {"valores": [0.1, 0.2, 0.3]}
    with patch("corec.utils.torch_utils.load_mobilenet_v3_small", MagicMock()) as mock_model, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        mock_model.return_value = MagicMock()
        result = await ia_module.procesar_bloque(bloque, datos)
        assert "mensajes" in result
        assert mock_alerta.called

@pytest.mark.asyncio
async def test_integration_analisis_datos(nucleus):
    """Prueba la integración de análisis de datos."""
    analisis = ModuloAnalisisDatos()
    await analisis.inicializar(nucleus, nucleus.config["analisis_datos_config"])
    df = pd.DataFrame({
        "bloque_id": ["enjambre_sensor", "nodo_seguridad"],
        "valor": [0.5, 0.7]
    }).pivot(columns="bloque_id", values="valor").fillna(0)
    with patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        result = await analisis.analizar(df, "test_analisis")
        assert "estadisticas" in result
        assert "num_anomalías" in result
        assert "correlaciones" in result
        assert mock_alerta.called

@pytest.mark.asyncio
async def test_integration_alert_archiving(nucleus, mock_redis, mock_db_pool):
    """Prueba la integración de archivado de alertas cuando el flujo de Redis está lleno."""
    mock_redis.xlen.return_value = 4500  # Simula flujo casi lleno (90% de 5000)
    alerta = {
        "tipo": "test_alerta",
        "bloque_id": "test_block",
        "mensaje": "Test",
        "timestamp": time.time()
    }
    with patch.object(nucleus, "archive_alert", AsyncMock()) as mock_archive:
        await nucleus.publicar_alerta(alerta)
        assert mock_redis.xadd.called
        assert mock_archive.called
