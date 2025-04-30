import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from corec.nucleus import CoreCNucleus
from plugins.registry import PluginRegistry

registry = PluginRegistry()

@pytest.mark.asyncio
async def test_plugin_load_valid(nucleus, tmp_path):
    """Prueba la carga de un complemento válido a través del registro."""
    plugin_config = {
        "enabled": True,
        "param": "value"
    }
    config_path = tmp_path / "test_plugin" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"test_plugin": plugin_config}))
    
    mock_plugin = MagicMock()
    mock_plugin.inicializar = AsyncMock()
    registry.plugins["test_plugin"] = mock_plugin
    with patch("importlib.import_module", return_value=MagicMock(test_plugin=mock_plugin)):
        await registry.load_plugin(nucleus, "test_plugin", {"enabled": True, "path": str(config_path)})
        assert mock_plugin.inicializar.called
    del registry.plugins["test_plugin"]

@pytest.mark.asyncio
async def test_plugin_load_invalid(nucleus):
    """Prueba la carga de un complemento no registrado."""
    config = {"enabled": True, "path": "fake/path/config.json"}
    with pytest.raises(ValueError, match="Complemento fake_plugin no está registrado"):
        await registry.load_plugin(nucleus, "fake_plugin", config)

@pytest.mark.asyncio
async def test_crypto_trading_plugin_load(nucleus, tmp_path):
    """Prueba la carga del plugin crypto_trading."""
    plugin_config = {
        "enabled": True,
        "api_key": "test_key",
        "api_secret": "test_secret",
        "symbols": ["BTC/USD"],
        "interval": "1h"
    }
    config_path = tmp_path / "crypto_trading" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"crypto_trading": plugin_config}))
    
    mock_plugin = MagicMock()
    mock_plugin.inicializar = AsyncMock()
    registry.plugins["crypto_trading"] = mock_plugin
    with patch("importlib.import_module", return_value=MagicMock(crypto_trading=mock_plugin)):
        await registry.load_plugin(nucleus, "crypto_trading", {"enabled": True, "path": str(config_path), "bloque": nucleus.config.bloques[0]})
        assert mock_plugin.inicializar.called
    del registry.plugins["crypto_trading"]
