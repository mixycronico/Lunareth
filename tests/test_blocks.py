# tests/test_blocks.py
import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad, crear_entidad
from corec.nucleus import CoreCNucleus

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
         patch("pandas.DataFrame", MagicMock()):  # Mock pandas para evitar dependencia
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

# Resto de las pruebas como en la versión anterior...
