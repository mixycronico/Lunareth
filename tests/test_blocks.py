import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad
from corec.entities_superpuestas import EntidadSuperpuesta
from corec.nucleus import CoreCNucleus

# Clase EntidadConError para simular un error al cambiar el estado
class EntidadConError(Entidad):
    def __init__(self, id: str, canal: int, procesar_func):
        self._estado = "inactiva"
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
        return {"valor": 0.5, "clasificacion": "test", "probabilidad": 0.9}

    entidades = [Entidad("ent_1", 1, mock_procesar)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    with patch.object(bloque.logger, "warning") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
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

    entidades = [Entidad("ent_1", 1, mock_procesar)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    with patch.object(bloque.logger, "warning") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
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

    entidades = [Entidad("ent_1", 1, mock_procesar)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    with patch.object(bloque.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
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

    entidades = [EntidadSuperpuesta("ent_1", {"rol1": 0.5, "rol2": 0.5}, quantization_step=0.05, min_fitness=0.3, mutation_rate=0.1, nucleus=nucleus)]
    entidades[0].estado = "inactiva"
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
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
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    with patch.object(bloque.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await bloque.reparar()  # No debería lanzar excepción, pero registra error
        assert mock_logger.called
        assert mock_alerta.called
        assert entidades[0].estado == "inactiva"

@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_exitoso(nucleus, mock_db_pool, monkeypatch):
    """Prueba la escritura exitosa en PostgreSQL."""
    async def mock_publicar_alerta(alerta):
        pass

    entidades = [Entidad("ent_1", 1, lambda: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    bloque.mensajes = [{"entidad_id": "ent_1", "canal": 1, "valor": 0.5, "timestamp": 12345, "roles": {}}]
    monkeypatch.setattr(nucleus, "publicar_alerta", mock_publicar_alerta)
    
    with patch.object(bloque.logger, "info") as mock_logger:
        await bloque.escribir_postgresql(mock_db_pool)
        assert len(bloque.mensajes) == 0
        assert mock_logger.called

@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_error(nucleus, mock_db_pool):
    """Prueba la escritura en PostgreSQL con un error y almacenamiento en fallback."""
    entidades = [Entidad("ent_1", 1, lambda carga: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    bloque.mensajes = [{
        "entidad_id": "ent_1",
        "canal": 1,
        "valor": 0.5,
        "clasificacion": "test",
        "probabilidad": 0.9,
        "timestamp": 12345,
        "roles": {}
    }]
    conn_mock = AsyncMock()
    conn_mock.execute.side_effect = Exception("DB Error")
    mock_db_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn_mock)
    mock_db_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    nucleus.db_pool = None  # Forzar el uso de fallback
    with patch.object(nucleus.logger, "warning") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta, \
         patch.object(nucleus, "save_fallback_messages", AsyncMock()) as mock_fallback, \
         patch("corec.modules.ejecucion.ModuloEjecucion.encolar_bloque", AsyncMock()) as mock_encolar:
        await nucleus.process_bloque(bloque)
        assert mock_logger.called
        assert mock_fallback.called
        assert mock_alerta.called
        assert mock_alerta.call_args[0][0]["tipo"] == "error_db_pool"
