# tests/test_modulo_ia.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from corec.modules.ia import ModuloIA
from corec.blocks import BloqueSimbiotico

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
    bloque.ia_timeout_seconds = 0.1  # Timeout muy corto
    datos = {"valores": [0.1, 0.2, 0.3]}
    with patch("corec.utils.torch_utils.load_mobilenet_v3_small", MagicMock()) as mock_model, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta, \
         patch("torch.cat", side_effect=lambda x, dim: asyncio.sleep(1)):
        result = await ia_module.procesar_bloque(bloque, datos)
        assert len(result["mensajes"]) == 1
        assert result["mensajes"][0]["clasificacion"] == "fallback"
        assert mock_alerta.called_with({"tipo": "timeout_ia", "bloque_id": "ia_analisis", "timeout": 0.1, "attempt": 3})

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
        assert mock_alerta.called_with({"tipo": "alerta_recursos", "bloque_id": "ia_analisis"})
