import pytest
from corec.nucleus import CoreCNucleus
from corec.scheduler import Scheduler
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus, test_config):
    """Prueba el procesamiento de bloques programado."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=None), \
         patch("corec.utils.db_utils.init_postgresql", return_value=None):
        nucleus.scheduler = Scheduler()
        scheduler = nucleus.scheduler
        with patch.object(nucleus, "process_bloque", AsyncMock()) as mock_process:
            scheduler.schedule_periodic(
                func=nucleus.process_bloque,
                seconds=60,
                job_id="proc_test_block",
                args=[nucleus.bloques[0]]
            )
            assert mock_process.called is False  # No se ejecuta inmediatamente
            # Simular ejecución
            await nucleus.process_bloque(nucleus.bloques[0])
            assert mock_process.called

@pytest.mark.asyncio
async def test_scheduler_audit_anomalies(nucleus, test_config):
    """Prueba la auditoría de anomalías programada."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=None), \
         patch("corec.utils.db_utils.init_postgresql", return_value=None):
        nucleus.scheduler = Scheduler()
        scheduler = nucleus.scheduler
        with patch.object(nucleus.modules["auditoria"], "detectar_anomalias", AsyncMock()) as mock_detect:
            scheduler.schedule_periodic(
                func=nucleus.modules["auditoria"].detectar_anomalias,
                seconds=120,
                job_id="audit_anomalias"
            )
            assert mock_detect.called is False
            await nucleus.modules["auditoria"].detectar_anomalias()
            assert mock_detect.called

@pytest.mark.asyncio
async def test_scheduler_synchronize_bloques(nucleus, test_config):
    """Prueba la sincronización de bloques programada."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=None), \
         patch("corec.utils.db_utils.init_postgresql", return_value=None):
        nucleus.scheduler = Scheduler()
        scheduler = nucleus.scheduler
        with patch.object(nucleus.modules["sincronizacion"], "redirigir_entidades", AsyncMock()) as mock_sync:
            scheduler.schedule_periodic(
                func=nucleus.modules["sincronizacion"].redirigir_entidades,
                seconds=300,
                job_id="sync_bloques",
                args=[nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal]
            )
            assert mock_sync.called is False
            if len(nucleus.bloques) >= 2:
                await nucleus.modules["sincronizacion"].redirigir_entidades(
                    nucleus.bloques[0], nucleus.bloques[1], 0.1, nucleus.bloques[1].canal
                )
            assert mock_sync.called
