import pytest
from corec.scheduler import Scheduler
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus):
    """Prueba el procesamiento de bloques programado."""
    scheduler = Scheduler(nucleus)
    with patch("corec.nucleus.CoreCNucleus.process_bloque", AsyncMock()) as mock_process:
        await scheduler.process_bloques()
        assert mock_process.called
        assert mock_process.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_audit_anomalies(nucleus):
    """Prueba la auditoría de anomalías programada."""
    scheduler = Scheduler(nucleus)
    with patch("corec.modules.auditoria.ModuloAuditoria.detectar_anomalias", AsyncMock()) as mock_detectar:
        await scheduler.audit_anomalies()
        assert mock_detectar.called
        assert mock_detectar.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_synchronize_bloques(nucleus):
    """Prueba la sincronización de bloques programada."""
    scheduler = Scheduler(nucleus)
    with patch("corec.modules.sincronizacion.ModuloSincronizacion.synchronize_bloques", AsyncMock()) as mock_synchronize:
        await scheduler.synchronize_bloques()
        assert mock_synchronize.called
        assert mock_synchronize.call_count >= 1
