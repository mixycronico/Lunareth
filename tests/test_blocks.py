import pytest
from unittest.mock import AsyncMock, MagicMock, patch  # Eliminamos 'call', ya que no se usa
from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad


# Nueva clase EntidadConError para simular un error al cambiar el estado


class EntidadConError(Entidad):
    def __init__(self, id: str, canal: int, procesar_func):
        # Evitamos que el constructor de Entidad intente establecer estado
        self._estado = "inactiva"  # Establecemos el estado directamente
        self.id = id
        self.canal = canal
        self.procesar_func = procesar_func

    @property
    def estado(self):
        return self._estado

    @estado.setter
    def estado(self, value):
        if value == "activa":
            raise Exception("Error al asignar estado")
        self._estado = value


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
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
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
    entidades[0].estado = "inactiva"
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    monkeypatch.setattr(nucleus, "publicar_alerta", mock_publicar_alerta)
    with patch.object(bloque.logger, "info") as mock_logger:
        await bloque.reparar()
        assert entidades[0].estado == "activa"
        assert bloque.fallos == 0
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_reparar_error(nucleus):
    """Prueba la reparación con un error."""
    entidades = [EntidadConError("ent_1", 1, lambda: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    with patch.object(bloque.logger, "error") as mock_logger, \
            patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        with pytest.raises(Exception):
            await bloque.reparar()
        assert mock_logger.called
        assert mock_alerta.called
        assert entidades[0].estado == "inactiva"


@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_exitoso(nucleus, mock_postgresql, monkeypatch):
    """Prueba la escritura exitosa en PostgreSQL."""
    async def mock_publicar_alerta(alerta):
        pass

    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    bloque.mensajes = [{"entidad_id": "ent_1", "canal": 1, "valor": 0.5, "timestamp": 12345}]
    monkeypatch.setattr(nucleus, "publicar_alerta", mock_publicar_alerta)
    # Configuramos el mock para simular un objeto de conexión síncrono
    conn = MagicMock()
    cursor = MagicMock()
    cursor.execute.return_value = None
    cursor.close.return_value = None
    conn.cursor.return_value = cursor
    conn.commit.return_value = None

    with patch.object(bloque.logger, "info") as mock_logger:
        await bloque.escribir_postgresql(conn)
        assert len(bloque.mensajes) == 0
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_error(nucleus, mock_postgresql, monkeypatch):
    """Prueba la escritura en PostgreSQL con un error."""
    async def mock_publicar_alerta(alerta):
        pass

    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    bloque.mensajes = [{"entidad_id": "ent_1", "canal": 1, "valor": 0.5, "timestamp": 12345}]
    monkeypatch.setattr(nucleus, "publicar_alerta", mock_publicar_alerta)
    # Configuramos el mock para simular un objeto de conexión síncrono que lanza una excepción
    conn = MagicMock()
    cursor = MagicMock()
    cursor.execute.side_effect = Exception("DB Error")
    cursor.close.return_value = None
    conn.cursor.return_value = cursor
    conn.commit.return_value = None

    with patch.object(bloque.logger, "error") as mock_logger:
        await bloque.escribir_postgresql(conn)
        assert mock_logger.called
