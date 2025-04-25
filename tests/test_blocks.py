# tests/test_blocks.py
import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad, crear_entidad
from corec.nucleus import CoreCNucleus
from typing import Dict

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

@pytest.fixture
async def nucleus(mock_redis, mock_db_pool, test_config):
    """Fixture para inicializar CoreCNucleus con mocks."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()) as mock_schedule, \
         patch("pandas.DataFrame", MagicMock()):  # Mock pandas para pruebas rápidas
        mock_schedule.return_value = None
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        yield nucleus
        await nucleus.detener()

@pytest.mark.asyncio
async def test_bloque_procesar_exitoso(nucleus, monkeypatch):
    """Prueba el procesamiento exitoso de un bloque simbiótico."""
    async def mock_procesar(carga):
        return {"valor": 0.5, "clasificacion": "test", "probabilidad": 0.9}

    entidades = [Entidad("ent_1", 1, lambda carga: {"valor": 0.5})]
    monkeypatch.setattr(entidades[0], "procesar", mock_procesar)
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
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

    entidades = [Entidad("ent_1", 1, lambda carga: {"valor": "invalid"})]
    monkeypatch.setattr(entidades[0], "procesar", mock_procesar)
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
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

    entidades = [Entidad("ent_1", 1, lambda carga: {"valor": 0.5})]
    monkeypatch.setattr(entidades[0], "procesar", mock_procesar)
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
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

    entidades = [Entidad("ent_1", 1, lambda carga: {"valor": 0.5})]
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
    entidades = [EntidadConError("ent_1", 1, lambda carga: {"valor": 0.5})]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    with patch.object(bloque.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        with pytest.raises(Exception):
            await bloque.reparar()
        assert mock_logger.called
        assert mock_alerta.called
        assert entidades[0].estado == "inactiva"

@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_exitoso(nucleus, mock_db_pool):
    """Prueba la escritura exitosa en PostgreSQL."""
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
    with patch.object(bloque.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await bloque.escribir_postgresql(mock_db_pool)
        assert len(bloque.mensajes) == 0
        assert mock_logger.called
        assert mock_alerta.called

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
    with patch.object(bloque.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta, \
         patch.object(nucleus, "save_fallback_messages", AsyncMock()) as mock_fallback:
        await nucleus.process_bloque(bloque)
        assert mock_logger.called
        assert mock_alerta.called
        assert mock_fallback.called
        mock_fallback.assert_called_with("test_block", bloque.mensajes)

@pytest.mark.asyncio
async def test_bloque_timeout_config(nucleus):
    """Prueba la configuración de ia_timeout_seconds en un bloque."""
    entidades = [Entidad("ent_1", 1, lambda carga: {"valor": 0.5})]
    bloque = BloqueSimbiotico("ia_analisis", 4, entidades, 50.0, nucleus)
    bloque.ia_timeout_seconds = 10.0
    assert bloque.ia_timeout_seconds == 10.0
    with patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        result = await bloque.procesar(0.5)
        assert result["bloque_id"] == "ia_analisis"
        assert mock_alerta.called
