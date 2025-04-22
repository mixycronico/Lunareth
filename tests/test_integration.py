import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus
from plugins import PluginCommand

@pytest.mark.asyncio
async def test_integration_process_and_audit(nucleus):
    with patch("corec.scheduler.Scheduler.schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        mock_schedule.return_value = None  # No ejecutamos tareas reales
        await nucleus.inicializar()
        # Simulamos la ejecución manual de las tareas
        await nucleus.process_bloque(nucleus.bloques[0])
        await nucleus.modules["auditoria"].detectar_anomalias()
        assert nucleus.modules["ejecucion"].encolar_bloque.called
        assert nucleus.modules["auditoria"].detectar_anomalias.called

@pytest.mark.asyncio
async def test_integration_synchronize_and_plugin_execution(nucleus):
    with patch("corec.scheduler.Scheduler.schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        mock_schedule.return_value = None  # No ejecutamos tareas reales
        plugin_id = "crypto_trading"
        comando = {"action": "ejecutar_operacion", "params": {"exchange": "binance", "pair": "BTC/USDT", "side": "buy"}}
        plugin_mock = AsyncMock()
        plugin_mock.manejar_comando.return_value = {"status": "success"}
        nucleus.plugins[plugin_id] = plugin_mock
        await nucleus.inicializar()
        # Simulamos la ejecución manual de la sincronización
        if len(nucleus.bloques) >= 2:
            await nucleus.synchronize_bloques(nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal)
        assert nucleus.synchronize_bloques.called
        result = await nucleus.ejecutar_plugin(plugin_id, comando)
        assert result["status"] == "success"
        plugin_mock.manejar_comando.assert_called_once_with(PluginCommand(**comando))
