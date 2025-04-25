# tests/test_blocks.py (partial)
import pytest
from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_error(nucleus, mock_db_pool):
    """Prueba la escritura en PostgreSQL con un error y almacenamiento en fallback."""
    entidades = [Entidad("ent_1", 1, lambda carga: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    bloque.mensajes = [{
        "entidad_id": "ent_1",
        "canal": 1,
        "valor": 0.5,
        "clasificacion": "test",
        "probabilidad": 0.9,
        "timestamp": 12345
    }]
    mock_db_pool.cursor.side_effect = Exception("DB Error")
    with patch.object(bloque.logger, "warning") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta, \
         patch.object(nucleus, "save_fallback_messages", AsyncMock()) as mock_fallback:
        await nucleus.process_bloque(bloque)
        assert mock_logger.called  # Check warning logger for fallback
        assert mock_fallback.called
