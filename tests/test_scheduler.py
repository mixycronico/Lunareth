import pytest
import asyncio
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus):
    nucleus.process_bloque = AsyncMock()
    await asyncio.sleep(1)  # Dar tiempo para que el scheduler ejecute
    assert nucleus.process_bloque.called
    assert nucleus.process_bloque.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_audit_anomalies(nucleus):
    nucleus.modules["auditoria"].detectar_anomalias = AsyncMock()
    await asyncio.sleep(1)  # Dar tiempo para que el scheduler ejecute
    assert nucleus.modules["auditoria"].detectar_anomalias.called
    assert nucleus.modules["auditoria"].detectar_anomalias.call_count >= 1

@pytest.mark.asyncio
async def test_scheduler_synchronize_bloques(nucleus):
    nucleus.synchronize_bloques = AsyncMock()
    await asyncio.sleep(1)  # Dar tiempo para que el scheduler ejecute
    assert nucleus.synchronize_bloques.called
    assert nucleus.synchronize_bloques.call_count >= 1
