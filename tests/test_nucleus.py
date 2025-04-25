# tests/test_nucleus.py
import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad

@pytest.fixture
async def nucleus(mock_redis, mock_db_pool, test_config):
    """Fixture para inicializar CoreCNucleus con mocks."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()) as mock_schedule, \
         patch("pandas.DataFrame", MagicMock()):
        mock_schedule.return_value = None
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        yield nucleus
        await nucleus.detener()

@pytest.mark.asyncio
async def test_nucleus_fallback_storage(test_config, mock_redis):
    """Prueba el almacenamiento en fallback cuando PostgreSQL falla."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", side_effect=Exception("DB Error")), \
         patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()), \
         patch("pandas.DataFrame", MagicMock()):
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        bloque = BloqueSimbiotico("enjambre_sensor", 1, [
            crear_entidad(f"ent_{i}", 1, lambda carga: {"valor": 0.5, "clasificacion": "test", "probabilidad": 0.9})
            for i in range(1)
        ], 1.0, nucleus)
        bloque.mensajes = [{
            "entidad_id": "ent_1",
            "canal": 1,
            "valor": 0.5,
            "clasificacion": "test",
            "probabilidad": 0.9,
            "timestamp": 12345
        }]
        await nucleus.process_bloque(bloque)
        fallback_file = Path("fallback_messages.json")
        assert fallback_file.exists()
        with open(fallback_file, "r") as f:
            messages = json.load(f)
        assert len(messages) == 1
        assert messages[0]["bloque_id"] == "enjambre_sensor"
        fallback_file.unlink()

@pytest.mark.asyncio
async def test_nucleus_retry_fallback(test_config, mock_redis, mock_db_pool):
    """Prueba el reintento de mensajes desde fallback a PostgreSQL."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()), \
         patch("pandas.DataFrame", MagicMock()):
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        fallback_file = Path("fallback_messages.json")
        messages = [{
            "bloque_id": "enjambre_sensor",
            "mensaje": {
                "entidad_id": "ent_1",
                "canal": 1,
                "valor": 0.5,
                "clasificacion": "test",
                "probabilidad": 0.9,
                "timestamp": 12345
            }
        }]
        with open(fallback_file, "w") as f:
            json.dump(messages, f)
        await nucleus.retry_fallback_messages()
        assert not fallback_file.exists()
        mock_db_pool.cursor().execute.assert_called()
