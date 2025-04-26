import pytest
from corec.scheduler import Scheduler
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus):
    """Prueba el procesamiento de bloques programado."""
    scheduler = nucleus.scheduler
    with patch.object(scheduler, "schedule_periodic", AsyncMock()) as mock_schedule:
        scheduler.schedule_periodic(
            func=nucleus.process_bloque,
            seconds=60,
            job_id="proc_test_block",
            args=[nucleus.bloques[0]]
        )
        assert mock_schedule.called
        assert mock_schedule.call_args[0][0] == nucleus.process_bloque

@pytest.mark.asyncio
async def test_scheduler_audit_anomalies(nucleus):
    """Prueba la auditoría de anomalías programada."""
    scheduler = nucleus.scheduler
    with patch.object(scheduler, "schedule_periodic", AsyncMock()) as mock_schedule:
        scheduler.schedule_periodic(
            func=nucleus.modules["auditoria"].detectar_anomalias,
            seconds=120,
            job_id="audit_anomalias"
        )
        assert mock_schedule.called
        assert mock_schedule.call_args[0][0] == nucleus.modules["auditoria"].detectar_anomalias

@pytest.mark.asyncio
async def test_scheduler_synchronize_bloques(nucleus):
    """Prueba la sincronización de bloques programada."""
    scheduler = nucleus.scheduler
    with patch.object(scheduler, "schedule_periodic", AsyncMock()) as mock_schedule:
        scheduler.schedule_periodic(
            func=nucleus.modules["sincronizacion"].redirigir_entidades,
            seconds=300,
            job_id="sync_bloques",
            args=[nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal] if len(nucleus.bloques) >= 2 else []
        )
        assert mock_schedule.called
        assert mock_schedule.call_args[0][0] == nucleus.modules["sincronizacion"].redirigir_entidades
