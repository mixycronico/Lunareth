import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus
from plugins import PluginCommand

@pytest.mark.asyncio
async def test_integration_process_and_audit(nucleus):
    with patch("corec.modules.ejecucion.ModuloEjecucion.encolar_bloque", new_callable=AsyncMock) as mock_encolar, \
         patch("corec.modules.auditoria.ModuloAuditoria.detectar_anomalias", new_callable=AsyncMock) as mock_detectar, \
         patch.object(nucleus.scheduler, "schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        async def execute_task(func, *args, **kwargs):
            await func(*args, **kwargs)
        mock_schedule.side_effect = [
            lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args)),
            lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args)),
            lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args))
        ]
        await nucleus.inicializar()
        await asyncio.sleep(2)
        assert mock_encolar.called
        assert mock_detectar.called

@pytest.mark.asyncio
async def test_integration_synchronize_and_plugin_execution(nucleus):
    with patch("corec.nucleus.CoreCNucleus.synchronize_bloques", new_callable=AsyncMock) as mock_synchronize, \
         patch.object(nucleus.scheduler, "schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        plugin_id = "crypto_trading"
        comando = {"action": "ejecutar_operacion", "params": {"exchange": "binance", "pair": "BTC/USDT", "side": "buy"}}
        plugin_mock = AsyncMock()
        plugin_mock.manejar_comando.return_value = {"status": "success"}
        nucleus.plugins[plugin_id] = plugin_mock
        async def execute_task(func, *args, **kwargs):
            await func(*args, **kwargs)
        mock_schedule.side_effect = [
            lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args)),
            lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args)),
            lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args))
        ]
        await nucleus.inicializar()
        await asyncio.sleep(2)
        assert mock_synchronize.called
        result = await nucleus.ejecutar_plugin(plugin_id, comando)
        assert result["status"] == "success"
        plugin_mock.manejar_comando.assert_called_once_with(PluginCommand(**comando))
