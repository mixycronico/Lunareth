import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad
from corec.nucleus import CoreCNucleus


@pytest.fixture
def mock_config():
    return {
        "db_config": {"host": "localhost"},
        "redis_config": {"host": "localhost", "port": 6379},
        "bloques": [],
        "plugins": {}
    }


@pytest.fixture
async def nucleus(mock_config):
    nucleus = CoreCNucleus("config.yml")
    with patch("corec.nucleus.cargar_config", return_value=mock_config):
        return nucleus


@pytest.mark.asyncio
async def test_bloque_procesar_exitoso(nucleus, monkeypatch):
    """Prueba el procesamiento exitoso de un bloque simbiótico."""
    async def mock_procesar(carga):
        return {"valor": 0.5}

    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    monkeypatch.setattr(entidades[0], "procesar", mock_procesar)
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    with patch.object(bloque.logger, "warning") as mock_logger, \
            patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        result = await bloque.procesar(0.5)
        assert result["bloque_id"] == "test_block"
        assert result["fitness"] == 0.5
        assert len(result["mensajes"]) == 1
        assert mock_alerta.called
        assert not mock_logger.called


@pytest.mark.asyncio
async def test_bloque_procesar_valor_invalido(nucleus, monkeypatch):
    """Prueba el procesamiento con un valor inválido."""
    async def mock_procesar(carga):
        return {"valor": "invalid"}

    entidades = [Entidad("ent_1", 1, lambda: {"valor": "invalid"})]
    monkeypatch.setattr(entidades[0], "procesar", mock_procesar)
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    with patch.object(bloque.logger, "warning") as mock_logger, \
            patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        result = await bloque.procesar(0.5)
        assert result["bloque_id"] == "test_block"
        assert result["fitness"] == 0
        assert len(result["mensajes"]) == 0
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_procesar_error_entidad(nucleus,
