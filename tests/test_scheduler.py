import pytest
import asyncio
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus):
    with patch.object(nucleus.scheduler, "schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        nucleus.process_bloque = AsyncMock()
        async def execute_task(func, *args, **kwargs):
            await func(*args, **kwargs)
        mock_schedule.side_effect = lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args))
        await nucleus.inicializar()
        await asyncio.sleep(2)
        assert nucleus.process_bloque.called
        assert nucleus.process_bloque.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_audit_anomalies(nucleus):
    with patch.object(nucleus.scheduler, "schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        nucleus.modules["auditoria"].detectar_anomalias = AsyncMock()
        async def execute_task(func, *args, **kwargs):
            await func(*args, **kwargs)
        mock_schedule.side_effect = lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args))
        await nucleus.inicializar()
        await asyncio.sleep(2)
        assert nucleus.modules["auditoria"].detectar_anomalias.called
        assert nucleus.modules["auditoria"].detectar_anomalias.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_synchronize_bloques(nucleus):
    with patch.object(nucleus.scheduler, "schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        nucleus.synchronize_bloques = AsyncMock()
        async def execute_task(func, *args, **kwargs):
            await func(*args, **kwargs)
        mock_schedule.side_effect = lambda func, *args, **kwargs: asyncio.create_task(execute_task(func, *args))
        await nucleus.inicializar()
        await asyncio.sleep(2)
        assert nucleus.synchronize_bloques.called
        assert nucleus.synchronize_bloques.call_count >= 1
