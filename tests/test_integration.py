import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus
from plugins import PluginCommand

@pytest.mark.asyncio
async def test_integration_process_and_audit(nucleus):
    with patch.object(nucleus.scheduler, "schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        bloque = nucleus.bloques[0]
        nucleus.modules["ejecucion"].encolar_bloque = AsyncMock()
        nucleus.modules["auditoria"].detectar_anomalias = AsyncMock()
        async def execute_task(func, *args, **kwargs):
            await func(*args, **kwargs)
        mock_schedule.side_effect = [
            lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args)),
            lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args)),
            lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args))
        ]
        await nucleus.inicializar()
        await asyncio.sleep(2)
        assert nucleus.modules["ejecucion"].encolar_bloque.called
        assert nucleus.modules["auditoria"].detectar_anomalias.called

@pytest.mark.asyncio
async def test_integration_synchronize_and_plugin_execution(nucleus):
    with patch.object(nucleus.scheduler, "schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        nucleus.synchronize_bloques = AsyncMock()
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
        assert nucleus.synchronize_bloques.called
        result = await nucleus.ejecutar_plugin(plugin_id, comando)
        assert result["status"] == "success"
        plugin_mock.manejar_comando.assert_called_once_with(PluginCommand(**comando))
