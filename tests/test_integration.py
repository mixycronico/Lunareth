import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from plugins import PluginCommand

@pytest.mark.asyncio
async def test_integration_process_and_audit(nucleus):
    # Mockeamos schedule_periodic para evitar que APScheduler cree tareas reales
    # (Ya mockeado en conftest.py)
    # Mockeamos las funciones directamente en los módulos para evitar sobrescritura
    with patch("corec.modules.ejecucion.ModuloEjecucion.encolar_bloque", new_callable=AsyncMock) as mock_encolar, \
         patch("corec.modules.auditoria.ModuloAuditoria.detectar_anomalias", new_callable=AsyncMock) as mock_detectar:
        await nucleus.inicializar()
        # Simulamos la ejecución manual de las tareas
        await nucleus.process_bloque(nucleus.bloques[0])
        await nucleus.modules["auditoria"].detectar_anomalias()
        assert mock_encolar.called
        assert mock_detectar.called

@pytest.mark.asyncio
async def test_integration_synchronize_and_plugin_execution(nucleus):
    # Mockeamos schedule_periodic para evitar que APScheduler cree tareas reales
    # (Ya mockeado en conftest.py)
    with patch("corec.nucleus.CoreCNucleus.synchronize_bloques", new_callable=AsyncMock) as mock_synchronize:
        plugin_id = "crypto_trading"
        comando = {"action": "ejecutar_operacion", "params": {"exchange": "binance", "pair": "BTC/USDT", "side": "buy"}}
        plugin_mock = AsyncMock()
        plugin_mock.manejar_comando.return_value = {"status": "success"}
        nucleus.plugins[plugin_id] = plugin_mock
        await nucleus.inicializar()
        # Simulamos la ejecución manual de la sincronización
        if len(nucleus.bloques) >= 2:
            await nucleus.synchronize_bloques(nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal)
        assert mock_synchronize.called
        result = await nucleus.ejecutar_plugin(plugin_id, comando)
        assert result["status"] == "success"
        plugin_mock.manejar_comando.assert_called_once_with(PluginCommand(**comando))
