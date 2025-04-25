# tests/test_module_ia.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from corec.modules.ia import ModuloIA
from corec.blocks import BloqueSimbiotico
from corec.nucleus import CoreCNucleus
from typing import Dict, Any

@pytest.fixture
async def nucleus(mock_redis, mock_db_pool, test_config):
    """Fixture para inicializar CoreCNucleus con mocks."""
    with patch("corec.config_loader.load_config_dict", return_value=test_config), \
         patch("corec.utils.db_utils.init_postgresql", return_value=mock_db_pool), \
         patch("corec.utils.db_utils.init_redis", return_value=mock_redis), \
         patch("corec.scheduler.Scheduler.schedule_periodic", AsyncMock()) as mock_schedule, \
         patch("pandas.DataFrame", MagicMock()), \
         patch("corec.utils.torch_utils.load_mobilenet_v3_small", MagicMock()) as mock_model:
        mock_schedule.return_value = None
        mock_model.return_value = MagicMock()
        nucleus = CoreCNucleus("config/corec_config.json")
        await nucleus.inicializar()
        yield nucleus
        await nucleus.detener()

@pytest.mark.asyncio
async def test_modulo_ia_inicializar(nucleus):
    """Prueba la inicialización de ModuloIA."""
    ia_module = ModuloIA()
    with patch("corec.utils.torch_utils.load_mobilenet_v3_small", MagicMock()) as mock_model, \
         patch.object(ia_module.logger, "info") as mock_logger:
        mock_model.return_value = MagicMock()
        await ia_module.inicializar(nucleus, nucleus.config["ia_config"])
        assert mock_logger.called
        assert ia_module.model is not None

@pytest.mark.asyncio
async def test_modulo_ia_procesar_timeout(nucleus):
    """Prueba el manejo de timeout en ModuloIA."""
    ia_module = ModuloIA()
    await ia_module.inicializar(nucleus, nucleus.config["ia_config"])
    bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus)
    bloque.ia_timeout_seconds = 0.1
    datos = {"valores": [0.1, 0.2, 0.3]}
    with patch("corec.utils.torch_utils.load_mobilenet_v3_small", MagicMock()) as mock_model, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta, \
         patch("torch.cat", side_effect=lambda x, dim: asyncio.sleep(1)):
        result = await ia_module.procesar_bloque(bloque, datos)
        assert len(result["mensajes"]) == 1
        assert result["mensajes"][0]["clasificacion"] == "fallback"
        assert mock_alerta.call_args[0][0]["tipo"] == "timeout_ia"
        assert mock_alerta.call_args[0][0]["bloque_id"] == "ia_analisis"
        assert mock_alerta.call_args[0][0]["timeout"] == 0.1
        assert mock_alerta.call_args[0][0]["attempt"] == 3

@pytest.mark.asyncio
async def test_modulo_ia_recursos_excedidos(nucleus):
    """Prueba el manejo de recursos excedidos en ModuloIA."""
    ia_module = ModuloIA()
    await ia_module.inicializar(nucleus, nucleus.config["ia_config"])
    bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus)
    datos = {"valores": [0.1, 0.2, 0.3]}
    with patch("corec.utils.torch_utils.load_mobilenet_v3_small", MagicMock()) as mock_model, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta, \
         patch("psutil.cpu_percent", return_value=95.0):
        result = await ia_module.procesar_bloque(bloque, datos)
        assert len(result["mensajes"]) == 1
        assert result["mensajes"][0]["clasificacion"] == "fallback_recursos"
        assert mock_alerta.call_args[0][0]["tipo"] == "alerta_recursos"
        assert mock_alerta.call_args[0][0]["bloque_id"] == "ia_analisis"
