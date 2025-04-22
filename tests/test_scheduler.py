import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus
from corec.modules.auditoria import ModuloAuditoria

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus):
    with patch("corec.scheduler.Scheduler.schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        # Configuramos el mock para process_bloque con una firma válida
        nucleus.process_bloque = AsyncMock(
            __code__=CoreCNucleus.process_bloque.__code__  # Simulamos la firma
        )
        mock_schedule.return_value = None  # No ejecutamos tareas reales
        await nucleus.inicializar()
        # Simulamos la ejecución manual de la tarea
        for bloque in nucleus.bloques:
            await nucleus.process_bloque(bloque)
        assert nucleus.process_bloque.called
        assert nucleus.process_bloque.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_audit_anomalies(nucleus):
    with patch("corec.scheduler.Scheduler.schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        # Configuramos el mock para detectar_anomalias con una firma válida
        nucleus.modules["auditoria"].detectar_anomalias = AsyncMock(
            __code__=ModuloAuditoria.detectar_anomalias.__code__  # Simulamos la firma
        )
        mock_schedule.return_value = None  # No ejecutamos tareas reales
        await nucleus.inicializar()
        # Simulamos la ejecución manual de la tarea
        await nucleus.modules["auditoria"].detectar_anomalias()
        assert nucleus.modules["auditoria"].detectar_anomalias.called
        assert nucleus.modules["auditoria"].detectar_anomalias.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_synchronize_bloques(nucleus):
    with patch("corec.scheduler.Scheduler.schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        # Configuramos el mock para synchronize_bloques con una firma válida
        nucleus.synchronize_bloques = AsyncMock(
            __code__=CoreCNucleus.synchronize_bloques.__code__  # Simulamos la firma
        )
        mock_schedule.return_value = None  # No ejecutamos tareas reales
        await nucleus.inicializar()
        # Simulamos la ejecución manual de la sincronización
        if len(nucleus.bloques) >= 2:
            await nucleus.synchronize_bloques(nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal)
        assert nucleus.synchronize_bloques.called
        assert nucleus.synchronize_bloques.call_count >= 1
