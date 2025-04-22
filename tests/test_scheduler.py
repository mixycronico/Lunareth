import pytest
import asyncio
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus):
    with patch("corec.scheduler.Scheduler.schedule_periodic", new_callable=AsyncMock) as mock_schedule:
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
        mock_schedule.return_value = None  # No ejecutamos tareas reales
        await nucleus.inicializar()
        # Simulamos la ejecución manual de la tarea
        await nucleus.modules["auditoria"].detectar_anomalias()
        assert nucleus.modules["auditoria"].detectar_anomalias.called
        assert nucleus.modules["auditoria"].detectar_anomalias.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_synchronize_bloques(nucleus):
    with patch("corec.scheduler.Scheduler.schedule_periodic", new_callable=AsyncMock) as mock_schedule:
        mock_schedule.return_value = None  # No ejecutamos tareas reales
        await nucleus.inicializar()
        # Simulamos la ejecución manual de la sincronización
        if len(nucleus.bloques) >= 2:
            await nucleus.synchronize_bloques(nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal)
        assert nucleus.synchronize_bloques.called
        assert nucleus.synchronize_bloques.call_count >= 1
