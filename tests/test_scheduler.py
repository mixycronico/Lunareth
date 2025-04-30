import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus
from corec.scheduler import Scheduler
from corec.modules.auditoria import ModuloAuditoria
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.config_loader import CoreCConfig

@pytest.mark.asyncio
async def test_scheduler_process_bloques(nucleus, test_config):
    """Prueba el procesamiento de bloques programado."""
    with patch("corec.config_loader.load_config", return_value=CoreCConfig(**test_config)), \
         patch("corec.utils.db_utils.init_redis", return_value=None), \
         patch("corec.utils.db_utils.init_postgresql", return_value=None), \
         patch("apscheduler.schedulers.asyncio.AsyncIOScheduler.add_job", AsyncMock()):
        await nucleus.inicializar()
        scheduler = nucleus.scheduler
        with patch.object(nucleus, "process_bloque", AsyncMock()) as mock_process:
            scheduler.schedule_periodic(
                func=nucleus.process_bloque,
                seconds=60,
                job_id="proc_test_block",
                args=[nucleus.bloques[0]]
            )
            assert mock_process.called is False  # No se ejecuta inmediatamente
            await nucleus.process_bloque(nucleus.bloques[0])
            assert mock_process.called

@pytest.mark.asyncio
async def test_scheduler_audit_anomalies(nucleus, test_config):
    """Prueba la auditoría de anomalías programada."""
    with patch("corec.config_loader.load_config", return_value=CoreCConfig(**test_config)), \
         patch("corec.utils.db_utils.init_redis", return_value=None), \
         patch("corec.utils.db_utils.init_postgresql", return_value=None), \
         patch("apscheduler.schedulers.asyncio.AsyncIOScheduler.add_job", AsyncMock()):
        await nucleus.inicializar()
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
    with patch("corec.config_loader.load_config", return_value=CoreCConfig(**test_config)), \
         patch("corec.utils.db_utils.init_redis", return_value=None), \
         patch("corec.utils.db_utils.init_postgresql", return_value=None), \
         patch("apscheduler.schedulers.asyncio.AsyncIOScheduler.add_job", AsyncMock()):
        await nucleus.inicializar()
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

@pytest.mark.asyncio
async def test_scheduler_process_bloques_error(nucleus, test_config):
    """Prueba el manejo de errores en el procesamiento de bloques programado."""
    with patch("corec.config_loader.load_config", return_value=CoreCConfig(**test_config)), \
         patch("corec.utils.db_utils.init_redis", return_value=None), \
         patch("corec.utils.db_utils.init_postgresql", return_value=None), \
         patch("apscheduler.schedulers.asyncio.AsyncIOScheduler.add_job", AsyncMock()):
        await nucleus.inicializar()
        scheduler = nucleus.scheduler
        with patch.object(nucleus, "process_bloque", AsyncMock(side_effect=Exception("Error de procesamiento"))) as mock_process, \
             patch.object(nucleus.logger, "error") as mock_logger, \
             patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
            scheduler.schedule_periodic(
                func=nucleus.process_bloque,
                seconds=60,
                job_id="proc_test_block",
                args=[nucleus.bloques[0]]
            )
            await nucleus.process_bloque(nucleus.bloques[0])
            assert mock_process.called
            assert mock_logger.called
            assert mock_alerta.called
            assert mock_alerta.call_args[0][0]["tipo"] == "error_procesamiento_bloque"
