import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus

@pytest.mark.asyncio
async def test_integration_process_and_audit(nucleus):
    with patch.object(nucleus.scheduler, "schedule_periodic", AsyncMock()) as mock_schedule:
        # Simular procesamiento de un bloque y auditoría
        bloque = nucleus.bloques[0]
        nucleus.modules["ejecucion"].encolar_bloque = AsyncMock()
        nucleus.modules["auditoria"].detectar_anomalias = AsyncMock()

        # Simular la ejecución de las tareas programadas
        mock_schedule.side_effect = [
            asyncio.create_task(nucleus.process_bloque(bloque)),
            asyncio.create_task(nucleus.modules["auditoria"].detectar_anomalias())
        ]
        await nucleus.inicializar()  # Re-inicializamos para aplicar el mock

        await asyncio.sleep(2)  # Dar tiempo para que las tareas se ejecuten

        assert nucleus.modules["ejecucion"].encolar_bloque.called
        assert nucleus.modules["auditoria"].detectar_anomalias.called

@pytest.mark.asyncio
async def test_integration_synchronize_and_plugin_execution(nucleus):
    with patch.object(nucleus.scheduler, "schedule_periodic", AsyncMock()) as mock_schedule:
        # Simular sincronización de bloques y ejecución de un plugin
        nucleus.synchronize_bloques = AsyncMock()
        plugin_id = "crypto_trading"
        comando = {"action": "ejecutar_operacion", "params": {"exchange": "binance", "pair": "BTC/USDT", "side": "buy"}}
        plugin_mock = AsyncMock()
        plugin_mock.manejar_comando.return_value = {"status": "success"}
        nucleus.plugins[plugin_id] = plugin_mock

        # Simular la ejecución de la tarea de sincronización
        mock_schedule.side_effect = [
            asyncio.create_task(nucleus.process_bloque(nucleus.bloques[0])),
            asyncio.create_task(nucleus.modules["auditoria"].detectar_anomalias()),
            asyncio.create_task(nucleus.synchronize_bloques(nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal))
        ]
        await nucleus.inicializar()  # Re-inicializamos para aplicar el mock

        await asyncio.sleep(2)  # Dar tiempo para que las tareas se ejecuten

        assert nucleus.synchronize_bloques.called

        result = await nucleus.ejecutar_plugin(plugin_id, comando)
        assert result["status"] == "success"
        plugin_mock.manejar_comando.assert_called_once_with(comando)
