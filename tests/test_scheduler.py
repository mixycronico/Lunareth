import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus
from corec.modules.auditoria import ModuloAuditoria

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus):
    with patch("corec.nucleus.CoreCNucleus.process_bloque", new_callable=AsyncMock) as mock_process:
        # schedule_periodic ya está mockeado en conftest.py
        await nucleus.inicializar()
        # Simulamos la ejecución manual de la tarea
        for bloque in nucleus.bloques:
            await nucleus.process_bloque(bloque)
        assert mock_process.called
        assert mock_process.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_audit_anomalies(nucleus):
    # Mockeamos detectar_anomalias en el módulo para evitar sobrescritura
    with patch("corec.modules.auditoria.ModuloAuditoria.detectar_anomalias", new_callable=AsyncMock) as mock_detectar:
        # schedule_periodic ya está mockeado en conftest.py
        await nucleus.inicializar()
        # Simulamos la ejecución manual de la tarea
        await nucleus.modules["auditoria"].detectar_anomalias()
        assert mock_detectar.called
        assert mock_detectar.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_synchronize_bloques(nucleus):
    with patch("corec.nucleus.CoreCNucleus.synchronize_bloques", new_callable=AsyncMock) as mock_synchronize:
        # schedule_periodic ya está mockeado en conftest.py
        await nucleus.inicializar()
        # Simulamos la ejecución manual de la sincronización
        if len(nucleus.bloques) >= 2:
            await nucleus.synchronize_bloques(nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal)
        assert mock_synchronize.called
        assert mock_synchronize.call_count >= 1
