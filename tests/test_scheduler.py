import pytest
import asyncio
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus):
    with patch.object(nucleus.scheduler, "schedule_periodic", AsyncMock()) as mock_schedule:
        nucleus.process_bloque = AsyncMock()
        mock_schedule.side_effect = lambda func, *args, **kwargs: asyncio.create_task(func(nucleus.bloques[0]))
        await nucleus.inicializar()  # Re-inicializamos para aplicar el mock
        await asyncio.sleep(2)  # Dar tiempo para que la tarea se ejecute
        assert nucleus.process_bloque.called
        assert nucleus.process_bloque.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_audit_anomalies(nucleus):
    with patch.object(nucleus.scheduler, "schedule_periodic", AsyncMock()) as mock_schedule:
        nucleus.modules["auditoria"].detectar_anomalias = AsyncMock()
        mock_schedule.side_effect = lambda func, *args, **kwargs: asyncio.create_task(func())
        await nucleus.inicializar()  # Re-inicializamos para aplicar el mock
        await asyncio.sleep(2)  # Dar tiempo para que la tarea se ejecute
        assert nucleus.modules["auditoria"].detectar_anomalias.called
        assert nucleus.modules["auditoria"].detectar_anomalias.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_synchronize_bloques(nucleus):
    with patch.object(nucleus.scheduler, "schedule_periodic", AsyncMock()) as mock_schedule:
        nucleus.synchronize_bloques = AsyncMock()
        mock_schedule.side_effect = lambda func, *args, **kwargs: asyncio.create_task(func(nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal))
        await nucleus.inicializar()  # Re-inicializamos para aplicar el mock
        await asyncio.sleep(2)  # Dar tiempo para que la tarea se ejecute
        assert nucleus.synchronize_bloques.called
        assert nucleus.synchronize_bloques.call_count >= 1
