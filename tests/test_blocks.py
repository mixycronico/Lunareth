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
    nucleus.redis_client = AsyncMock()  # Mockeamos redis_client para que publicar_alerta funcione
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
async def test_bloque_procesar_error_entidad(nucleus, monkeypatch):
    """Prueba el procesamiento con un error en una entidad."""
    async def mock_procesar(carga):
        raise Exception("Error")

    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    monkeypatch.setattr(entidades[0], "procesar", mock_procesar)
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10. días, nucleus)
    with patch.object(bloque.logger, "error") as mock_logger, \
            patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        result = await bloque.procesar(0.5)
        assert result["bloque_id"] == "test_block"
        assert result["fitness"] == 0
        assert len(result["mensajes"]) == 0
        assert bloque.fallos == 1
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_reparar_exitoso(nucleus, monkeypatch):
    """Prueba la reparación exitosa de un bloque."""
    async def mock_publicar_alerta(alerta):
        pass

    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    # Agregamos el atributo estado manualmente
    entidades[0].estado = "inactiva"
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    monkeypatch.setattr(nucleus, "publicar_alerta", mock_publicar_alerta)
    with patch.object(bloque.logger, "info") as mock_logger:
        await bloque.reparar()
        assert entidades[0].estado == "activa"
        assert bloque.fallos == 0
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_reparar_error(nucleus, monkeypatch):
    """Prueba la reparación con un error."""
    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    # Agregamos el atributo estado manualmente
    entidades[0].estado = "inactiva"
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    with patch.object(bloque.logger, "error") as mock_logger, \
            patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        # Simulamos un error al intentar modificar el estado
        def raise_error(value):
            raise Exception("Error")
        monkeypatch.setattr(entidades[0], "estado", property(lambda self: "inactiva", raise_error))
        with pytest.raises(Exception):  # Capturamos la excepción relanzada
            await bloque.reparar()
        assert mock_logger.called
        assert mock_alerta.called
        assert entidades[0].estado == "inactiva"  # Verificamos que el estado no cambió debido al error


@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_exitoso(nucleus, mock_postgresql, monkeypatch):
    """Prueba la escritura exitosa en PostgreSQL."""
    async def mock_publicar_alerta(alerta):
        pass

    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    bloque.mensajes = [{"entidad_id": "ent_1", "canal": 1, "valor": 0.5, "timestamp": 12345}]
    monkeypatch.setattr(nucleus, "publicar_alerta", mock_publicar_alerta)
    mock_cursor = MagicMock()
    mock_postgresql.cursor.return_value = mock_cursor
    with patch.object(bloque.logger, "info") as mock_logger:
        await bloque.escribir_postgresql(mock_postgresql)
        assert len(bloque.mensajes) == 0
        assert mock_logger.called
        assert mock_cursor.execute.called
        assert mock_postgresql.commit.called


@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_error(nucleus, mock_postgresql, monkeypatch):
    """Prueba la escritura en PostgreSQL con un error."""
    async def mock_publicar_alerta(alerta):
        pass

    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    bloque.mensajes = [{"entidad_id": "ent_1", "canal": 1, "valor": 0.5, "timestamp": 12345}]
    monkeypatch.setattr(nucleus, "publicar_alerta", mock_publicar_alerta)
    with patch.object(bloque.logger, "error") as mock_logger, \
            patch.object(mock_postgresql, "cursor", side_effect=Exception("DB Error")):
        await bloque.escribir_postgresql(mock_postgresql)
        assert mock_logger.called
