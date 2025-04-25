# tests/test_plugin.py
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from plugins.registry import registry
from corec.nucleus import CoreCNucleus

@pytest.mark.asyncio
async def test_plugin_load_valid(nucleus):
    """Prueba la carga de un complemento válido a través del registro."""
    config = {"test_plugin": {"param": "value"}}
    with patch("importlib.import_module") as mock_import:
        mock_plugin = MagicMock()
        mock_plugin.inicializar = AsyncMock()
        mock_import.return_value = mock_plugin
        await registry.load_plugin(nucleus, "test_plugin", config)
        mock_plugin.inicializar.assert_called_with(nucleus, config)
        assert nucleus.logger.info.called

@pytest.mark.asyncio
async def test_plugin_load_invalid(nucleus):
    """Prueba la carga de un complemento no registrado."""
    config = {"fake_plugin": {"param": "value"}}
    with pytest.raises(ValueError, match="Complemento fake_plugin no está registrado"):
        await registry.load_plugin(nucleus, "fake_plugin", config)
    assert nucleus.logger.error.called
