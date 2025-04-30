import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from corec.modules.ia import ModuloIA
from corec.blocks import BloqueSimbiotico
from corec.nucleus import CoreCNucleus
from torchvision.models import mobilenet_v3_small
from pathlib import Path

@pytest.mark.asyncio
async def test_modulo_ia_inicializar(nucleus, test_config):
    """Prueba la inicialización de ModuloIA."""
    ia_module = ModuloIA()
    with patch.object(ia_module.logger, "info") as mock_logger, \
         patch("pathlib.Path.exists", return_value=True), \
         patch("corec.utils.torch_utils.load_mobilenet_v3_small", AsyncMock()) as mock_load_model:
        config = test_config["ia_config"].copy()
        config["enabled"] = True
        config["model_path"] = "corec/models/mobilev3/model.pth"
        await ia_module.inicializar(nucleus, config)
        assert mock_logger.called
        assert mock_load_model.called

@pytest.mark.asyncio
async def test_modulo_ia_inicializar_modelo_no_encontrado(nucleus, test_config):
    """Prueba la inicialización de ModuloIA con modelo no encontrado."""
    ia_module = ModuloIA()
    with patch.object(ia_module.logger, "error") as mock_logger, \
         patch("pathlib.Path.exists", return_value=False), \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        config = test_config["ia_config"].copy()
        config["enabled"] = True
        config["model_path"] = "corec/models/mobilev3/model.pth"
        with pytest.raises(FileNotFoundError, match="Modelo no encontrado"):
            await ia_module.inicializar(nucleus, config)
        assert mock_logger.called
        assert mock_alerta.called

@pytest.mark.asyncio
async def test_modulo_ia_procesar_timeout(nucleus, test_config):
    """Prueba el manejo de timeout en ModuloIA."""
    ia_module = ModuloIA()
    config = test_config["ia_config"].copy()
    config["enabled"] = True
    config["model_path"] = "corec/models/mobilev3/model.pth"
    mock_model = MagicMock()
    def delayed_execution(x):
        time.sleep(1)  # Simular retraso síncrono
        return torch.zeros(1, 3)
    mock_model.side_effect = delayed_execution
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 3)
    dummy_state_dict = model.state_dict()
    with patch("pathlib.Path.exists", return_value=True), \
         patch("torch.load", return_value=dummy_state_dict), \
         patch("corec.utils.torch_utils.load_mobilenet_v3_small", return_value=mock_model), \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await ia_module.inicializar(nucleus, config)
        bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus, quantization_step=0.05, max_errores=0.1)
        bloque.ia_timeout_seconds = 0.01  # Timeout muy corto para forzar fallo
        datos = {"valores": [0.0] * (224 * 224 * 3)}  # Tamaño correcto
        result = await ia_module.procesar_bloque(bloque, datos)
        assert len(result["mensajes"]) == 1
        assert result["mensajes"][0]["clasificacion"] == "fallback"
        assert mock_alerta.called
        assert mock_alerta.call_args[0][0]["tipo"] == "timeout_ia"

@pytest.mark.asyncio
async def test_modulo_ia_recursos_excedidos(nucleus, test_config):
    """Prueba el manejo de recursos excedidos en ModuloIA."""
    ia_module = ModuloIA()
    config = test_config["ia_config"].copy()
    config["enabled"] = True
    config["model_path"] = "corec/models/mobilev3/model.pth"
    config["max_size_mb"] = 50
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 3)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 3)
    dummy_state_dict = model.state_dict()
    with patch("pathlib.Path.exists", return_value=True), \
         patch("torch.load", return_value=dummy_state_dict), \
         patch("corec.utils.torch_utils.load_mobilenet_v3_small", return_value=mock_model), \
         patch("psutil.cpu_percent", return_value=95.0), \
         patch("psutil.virtual_memory", return_value=MagicMock(used=100*1024*1024)), \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await ia_module.inicializar(nucleus, config)
        bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus, quantization_step=0.05, max_errores=0.1)
        datos = {"valores": [0.0] * (224 * 224 * 3)}  # Tamaño correcto
        result = await ia_module.procesar_bloque(bloque, datos)
        assert len(result["mensajes"]) == 1
        assert result["mensajes"][0]["clasificacion"] == "fallback_recursos"
        assert mock_alerta.call_args[0][0]["tipo"] == "alerta_recursos"

@pytest.mark.asyncio
async def test_modulo_ia_datos_invalidos(nucleus, test_config):
    """Prueba el manejo de datos inválidos en ModuloIA."""
    ia_module = ModuloIA()
    config = test_config["ia_config"].copy()
    config["enabled"] = True
    config["model_path"] = "corec/models/mobilev3/model.pth"
    mock_model = MagicMock()
    mock_model.return_value = torch.zeros(1, 3)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 3)
    dummy_state_dict = model.state_dict()
    with patch("pathlib.Path.exists", return_value=True), \
         patch("torch.load", return_value=dummy_state_dict), \
         patch("corec.utils.torch_utils.load_mobilenet_v3_small", return_value=mock_model), \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await ia_module.inicializar(nucleus, config)
        bloque = BloqueSimbiotico("ia_analisis", 4, [], 50.0, nucleus, quantization_step=0.05, max_errores=0.1)
        datos = {"valores": [0.1, 0.2]}  # Tamaño incorrecto
        result = await ia_module.procesar_bloque(bloque, datos)
        assert len(result["mensajes"]) == 1
        assert result["mensajes"][0]["clasificacion"] == "error_datos"
        assert mock_alerta.called
        assert mock_alerta.call_args[0][0]["tipo"] == "error_datos_entrada"
