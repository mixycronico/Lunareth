import pytest
import asyncio
from unittest.mock import AsyncMock
from corec.nucleus import CoreCNucleus

@pytest.mark.asyncio
async def test_integration_process_and_audit(nucleus):
    # Simular procesamiento de un bloque y auditoría
    bloque = nucleus.bloques[0]
    nucleus.modules["ejecucion"].encolar_bloque = AsyncMock()
    nucleus.modules["auditoria"].detectar_anomalias = AsyncMock()

    await asyncio.sleep(1)  # Dar tiempo para que el scheduler ejecute

    assert nucleus.modules["ejecucion"].encolar_bloque.called
    assert nucleus.modules["auditoria"].detectar_anomalias.called

@pytest.mark.asyncio
async def test_integration_synchronize_and_plugin_execution(nucleus):
    # Simular sincronización de bloques y ejecución de un plugin
    nucleus.synchronize_bloques = AsyncMock()
    plugin_id = "crypto_trading"
    comando = {"action": "ejecutar_operacion", "params": {"exchange": "binance", "pair": "BTC/USDT", "side": "buy"}}
    plugin_mock = AsyncMock()
    plugin_mock.manejar_comando.return_value = {"status": "success"}
    nucleus.plugins[plugin_id] = plugin_mock

    await asyncio.sleep(1)  # Dar tiempo para que el scheduler ejecute

    assert nucleus.synchronize_bloques.called

    result = await nucleus.ejecutar_plugin(plugin_id, comando)
    assert result["status"] == "success"
    plugin_mock.manejar_comando.assert_called_once_with(comando)
