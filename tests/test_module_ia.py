import pytest
from corec.modules.ia import ModuloIA
from corec.blocks import BloqueSimbiotico
from unittest.mock import AsyncMock, patch, MagicMock
import torch
import asyncio
import numpy as np

@pytest.mark.asyncio
async def test_modulo_ia_inicializar(nucleus):
    """Prueba la inicialización de ModuloIA."""
    ia_module = ModuloIA()
    with patch.object(ia_module.logger, "info") as mock_logger:
        await ia_module.inicializar(nucleus, nucleus.config["ia_config"])
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_ia_procesar_timeout(nucleus):
    """Prueba el manejo de timeout en ModuloIA."""
    ia_module = ModuloIA()
    config = nucleus.config["ia_config"].copy()
    config["enabled"] = True
    await ia_module.inicializar(nucleus, config)
    bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus)
    bloque.ia_timeout_seconds = 0.1
    datos = {"valores": [0.1, 0.2, 0.3]}
    mock_model = MagicMock()
    mock_model.side_effect = lambda x: asyncio.sleep(1)
    with patch("corec.utils.torch_utils.load_mobilenet_v3_small", return_value=mock_model), \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        result = await ia_module.procesar_bloque(bloque, datos)
        assert len(result["mensajes"]) == 1
        assert result["mensajes"][0]["clasificacion"] == "fallback"
        assert mock_alerta.called
        assert mock_alerta.call_args[0][0]["tipo"] == "timeout_ia"

@pytest.mark.asyncio
async def test_modulo_ia_recursos_excedidos(nucleus):
    """Prueba el manejo de recursos excedidos en ModuloIA."""
    ia_module = ModuloIA()
    config = nucleus.config["ia_config"].copy()
    config["enabled"] = True
    await ia_module.inicializar(nucleus, config)
    bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus)
    datos = {"valores": [0.1, 0.2, 0.3]}
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 3)  # Simular logits válidos
    with patch("corec.utils.torch_utils.load_mobilenet_v3_small", return_value=mock_model), \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta, \
         patch("psutil.cpu_percent", return_value=95.0):
        result = await ia_module.procesar_bloque(bloque, datos)
        assert len(result["mensajes"]) == 1
        assert result["mensajes"][0]["clasificacion"] == "fallback_recursos"
        assert mock_alerta.called
        assert mock_alerta.call_args[0][0]["tipo"] == "alerta_recursos"
