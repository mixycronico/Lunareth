import pytest
from corec.nucleus import CoreCNucleus
from unittest.mock import AsyncMock, patch, MagicMock
import json
from pathlib import Path

@pytest.mark.asyncio
async def test_nucleus_fallback_storage(test_config, mock_redis, mock_db_pool):
    """Prueba el almacenamiento en fallback cuando PostgreSQL falla."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()), \
         patch("pandas.DataFrame", MagicMock()), \
         patch("corec.utils.torch_utils.load_mobilenet_v3_small", MagicMock()) as mock_model:
        mock_model.return_value = MagicMock()
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        messages = [{
            "entidad_id": "ent_1",
            "canal": 1,
            "valor": 0.5,
            "clasificacion": "test",
            "probabilidad": 0.9,
            "timestamp": 12345
        }]
        with patch("json.dump", MagicMock()) as mock_json_dump:
            await nucleus.save_fallback_messages(bloque_id="test_block", mensajes=messages)
            assert mock_json_dump.called
        await nucleus.detener()

@pytest.mark.asyncio
async def test_nucleus_retry_fallback(test_config, mock_redis, mock_db_pool, tmp_path):
    """Prueba el reintento de mensajes desde fallback a PostgreSQL."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()), \
         patch("pandas.DataFrame", MagicMock()), \
         patch("corec.utils.torch_utils.load_mobilenet_v3_small", MagicMock()) as mock_model:
        mock_model.return_value = MagicMock()
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        fallback_file = tmp_path / "fallback_messages.json"
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
        mock_db_pool.execute = AsyncMock(return_value=None)
        nucleus.fallback_storage = fallback_file  # Ajustar path para pruebas
        await nucleus.retry_fallback_messages()
        assert not fallback_file.exists()
        assert mock_db_pool.execute.called
        await nucleus.detener()
