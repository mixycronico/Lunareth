import pytest
from corec.modules.ia import ModuloIA
from corec.blocks import BloqueSimbiotico
from unittest.mock import AsyncMock, patch, MagicMock
import torch
import asyncio
import numpy as np
import time

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
    config["model_path"] = "corec/models/mobilev3/model.pth"
    mock_model = MagicMock()
    def delayed_execution(x):
        time.sleep(1)  # Simular retraso síncrono
        return torch.zeros(1, 3)
    mock_model.side_effect = delayed_execution
    with patch("torch.load", return_value={}), \
         patch("corec.utils.torch_utils.load_mobilenet_v3_small", return_value=mock_model), \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await ia_module.inicializar(nucleus, config)
        bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus)
        bloque.ia_timeout_seconds = 0.1
        datos = {"valores": [0.1, 0.2, 0.3]}
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
    config["model_path"] = "corec/models/mobilev3/model.pth"
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 3)
    with patch("torch.load", return_value={}), \
         patch("corec.utils.torch_utils.load_mobilenet_v3_small", return_value=mock_model), \
         patch("psutil.cpu_percent", return_value=95.0), \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await ia_module.inicializar(nucleus, config)
        bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus)
        datos = {"valores": [0.1, 0.2, 0.3]}
        result = await ia_module.procesar_bloque(bloque, datos)
        assert len(result["mensajes"]) == 1
        assert result["mensajes"][0]["clasificacion"] == "fallback_recursos"
        assert mock_alerta.called
        assert mock_alerta.call_args[0][0]["tipo"] == "alerta_recursos"
