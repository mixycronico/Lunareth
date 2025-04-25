# tests/test_plugin.py
import pytest
from plugins.registry import PluginRegistry
from unittest.mock import AsyncMock, patch, MagicMock

registry = PluginRegistry()

@pytest.mark.asyncio
async def test_plugin_load_valid(nucleus):
    """Prueba la carga de un complemento válido a través del registro."""
    config = {"test_plugin": {"param": "value"}}
    # Register a mock plugin
    mock_plugin = MagicMock()
    mock_plugin.inicializar = AsyncMock()
    registry.plugins["test_plugin"] = mock_plugin
    with patch("importlib.import_module", return_value=mock_plugin):
        await registry.load_plugin(nucleus, "test_plugin", config)
        assert mock_plugin.inicializar.called
        # Clean up
        del registry.plugins["test_plugin"]

@pytest.mark.asyncio
async def test_plugin_load_invalid(nucleus):
    """Prueba la carga de un complemento no registrado."""
    config = {"fake_plugin": {"param": "value"}}
    with pytest.raises(ValueError, match="Complemento fake_plugin no está registrado"), \
         patch.object(nucleus.logger, "error") as mock_logger:
        await registry.load_plugin(nucleus, "fake_plugin", config)
    assert mock_logger.called
