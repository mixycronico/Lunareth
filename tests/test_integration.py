import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus
from corec.blocks import BloqueSimbiotico
from corec.modules.ia import ModuloIA
from corec.modules.analisis_datos import ModuloAnalisisDatos
import pandas as pd
from pathlib import Path
import json

@pytest.mark.asyncio
async def test_integration_process_and_audit(nucleus):
    """Prueba la integración de procesamiento y auditoría."""
    with patch("corec.modules.ejecucion.ModuloEjecucion.encolar_bloque", AsyncMock()) as mock_encolar, \
         patch("corec.modules.auditoria.ModuloAuditoria.detectar_anomalias", AsyncMock()) as mock_detectar, \
         patch("apscheduler.schedulers.asyncio.AsyncIOScheduler.add_job", AsyncMock()):
        await nucleus.inicializar()
        await nucleus.process_bloque(nucleus.bloques[0])
        await nucleus.modules["auditoria"].detectar_anomalias()
        assert mock_encolar.called
        assert mock_detectar.called

@pytest.mark.asyncio
async def test_integration_synchronize_and_plugin_execution(nucleus, tmp_path):
    """Prueba la integración de sincronización y ejecución de plugins."""
    # Crear archivo de configuración para crypto_trading
    plugin_config = {
        "crypto_trading": {
            "enabled": True,
            "api_key": "test_key",
            "api_secret": "test_secret",
            "symbols": ["BTC/USD"],
            "interval": "1h"
        }
    }
    config_path = tmp_path / "crypto_trading" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(plugin_config))

    with patch("corec.modules.sincronizacion.ModuloSincronizacion.redirigir_entidades", AsyncMock()) as mock_synchronize, \
         patch("corec.config_loader.load_config", return_value=nucleus.config) as mock_load_config:
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
    config = test_config["ia_config"].copy()
    config["enabled"] = True
    config["model_path"] = "corec/models/mobilev3/model.pth"
    bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus, quantization_step=0.05, max_errores=0.1)
    datos = {"valores": [0.0] * (224 * 224 * 3)}  # Tamaño correcto
    with patch("corec.utils.torch_utils.load_mobilenet_v3_small", AsyncMock()) as mock_load_model, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.1, 0.8, 0.1]])
        mock_load_model.return_value = mock_model
        await ia_module.inicializar(nucleus, config)
        result = await ia_module.procesar_bloque(bloque, datos)
        assert "mensajes" in result
        assert len(result["mensajes"]) == 1
        assert result["mensajes"][0]["clasificacion"] == "clase_1"
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
    mock_redis.xlen.return_value = 9000  # Simular flujo lleno
    alerta = {
        "tipo": "test_alerta",
        "bloque_id": "test_block",
        "mensaje": "Test",
        "timestamp": time.time()
    }
    with patch.object(nucleus, "archive_alert", AsyncMock()) as mock_archive:
        await nucleus.publicar_alerta(alerta)
        assert mock_archive.called

@pytest.mark.asyncio
async def test_integration_celery_task_execution(nucleus, test_config):
    """Prueba la integración con Celery para tareas asíncronas."""
    with patch("corec.modules.ejecucion.ModuloEjecucion.encolar_bloque", AsyncMock()) as mock_encolar, \
         patch("corec.celery_config.celery_app.send_task", AsyncMock()) as mock_send_task:
        await nucleus.inicializar()
        bloque = BloqueSimbiotico("test_block", 1, [], 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
        await nucleus.modules["ejecucion"].encolar_bloque(bloque)
        assert mock_encolar.called
        assert mock_send_task.called
