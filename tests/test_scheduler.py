import pytest
import asyncio
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus):
    with patch("corec.nucleus.CoreCNucleus.process_bloque", new_callable=AsyncMock) as mock_process, \
         patch.object(nucleus.scheduler, "schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        async def execute_task(func, *args, **kwargs):
            await func(*args, **kwargs)
        mock_schedule.side_effect = lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args))
        await nucleus.inicializar()
        await asyncio.sleep(2)
        assert mock_process.called
        assert mock_process.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_audit_anomalies(nucleus):
    with patch("corec.modules.auditoria.ModuloAuditoria.detectar_anomalias", new_callable=AsyncMock) as mock_detectar, \
         patch.object(nucleus.scheduler, "schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        async def execute_task(func, *args, **kwargs):
            await func(*args, **kwargs)
        mock_schedule.side_effect = lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args))
        await nucleus.inicializar()
        await asyncio.sleep(2)
        assert mock_detectar.called
        assert mock_detectar.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_synchronize_bloques(nucleus):
    with patch("corec.nucleus.CoreCNucleus.synchronize_bloques", new_callable=AsyncMock) as mock_synchronize, \
         patch.object(nucleus.scheduler, "schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        async def execute_task(func, *args, **kwargs):
            await func(*args, **kwargs)
        mock_schedule.side_effect = lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args))
        await nucleus.inicializar()
        await asyncio.sleep(2)
        assert mock_synchronize.called
        assert mock_synchronize.call_count >= 1
