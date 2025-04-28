import pytest
from corec.nucleus import CoreCNucleus
from unittest.mock import AsyncMock, patch, MagicMock
import json
from pathlib import Path
import pandas as pd

@pytest.mark.asyncio
async def test_nucleus_fallback_storage(test_config, mock_redis, mock_db_pool):
    """Prueba el almacenamiento en fallback cuando PostgreSQL falla."""
    test_config["ia_config"]["enabled"] = False
    test_config["ia_config"]["model_path"] = ""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()), \
         patch("pandas.DataFrame", return_value=pd.DataFrame({"valores": [0.1, 0.2, 0.3]}, dtype=float)):
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
    test_config["ia_config"]["enabled"] = False
    test_config["ia_config"]["model_path"] = ""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()), \
         patch("pandas.DataFrame", return_value=pd.DataFrame({"valores": [0.1, 0.2, 0.3]}, dtype=float)):
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        nucleus.db_pool = mock_db_pool  # Asegurar que db_pool esté configurado
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
        conn = AsyncMock()
        conn.execute = AsyncMock(return_value=None)
        mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
        mock_db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        nucleus.fallback_storage = fallback_file
        nucleus.logger.info(f"[Debug] Attempting to retry fallback messages from {fallback_file}")
        await nucleus.retry_fallback_messages()
        nucleus.logger.info(f"[Debug] File exists after retry: {fallback_file.exists()}")
        assert not fallback_file.exists()
        assert conn.execute.called
        assert conn.execute.call_args[0][0].startswith("INSERT INTO mensajes")
