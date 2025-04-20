import pytest
from unittest.mock import patch, AsyncMock
from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad


@pytest.mark.asyncio
async def test_bloque_procesar_exitoso(nucleus):
    """Prueba el procesamiento exitoso de un bloque simbiótico."""
    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
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
async def test_bloque_procesar_valor_invalido(nucleus):
    """Prueba el procesamiento con un valor inválido."""
    entidades = [Entidad("ent_1", 1, lambda: {"valor": "invalid"})]
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
async def test_bloque_procesar_error_entidad(nucleus):
    """Prueba el procesamiento con un error en una entidad."""
    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    with patch.object(bloque.logger, "error") as mock_logger, \
            patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta, \
            patch.object(Entidad, "procesar", side_effect=Exception("Error")):
        result = await bloque.procesar(0.5)
        assert result["bloque_id"] == "test_block"
        assert result["fitness"] == 0
        assert len(result["mensajes"]) == 0
        assert bloque.fallos == 1
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_reparar_exitoso(nucleus):
    """Prueba la reparación exitosa de un bloque."""
    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    entidades[0].estado = "inactiva"
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    with patch.object(bloque.logger, "info") as mock_logger, \
            patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await bloque.reparar()
        assert entidades[0].estado == "activa"
        assert bloque.fallos == 0
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_reparar_error(nucleus):
    """Prueba la reparación con un error."""
    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    with patch.object(bloque.logger, "error") as mock_logger, \
            patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta, \
            patch.object(Entidad, "estado", side_effect=Exception("Error")):
        await bloque.reparar()
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_exitoso(nucleus, mock_postgresql):
    """Prueba la escritura exitosa en PostgreSQL."""
    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    bloque.mensajes = [{"entidad_id": "ent_1", "canal": 1, "valor": 0.5, "timestamp": 12345}]
    with patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await bloque.escribir_postgresql(mock_postgresql)
        assert len(bloque.mensajes) == 0
        assert mock_alerta.called


@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_error(nucleus, mock_postgresql):
    """Prueba la escritura en PostgreSQL con un error."""
    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    bloque.mensajes = [{"entidad_id": "ent_1", "canal": 1, "valor": 0.5, "timestamp": 12345}]
    with patch.object(bloque.logger, "error") as mock_logger, \
            patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta, \
            patch.object(mock_postgresql, "cursor", side_effect=Exception("DB Error")):
        await bloque.escribir_postgresql(mock_postgresql)
        assert mock_alerta.called
        assert mock_logger.called
