import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from plugins.example_plugin.main import ExamplePlugin


@pytest.mark.asyncio
async def test_plugin_inicializar(nucleus):
    """Prueba la inicialización de ExamplePlugin."""
    plugin = ExamplePlugin()
    config = {}  # Configuración vacía
    with patch.object(plugin.logger, "info") as mock_logger:
        await plugin.inicializar(nucleus, config)
        assert plugin.nucleus == nucleus
        assert mock_logger.called


@pytest.mark.asyncio
async def test_plugin_manejar_comando(nucleus):
    """Prueba el manejo de comandos en ExamplePlugin."""
    plugin = ExamplePlugin()
    config = {}  # Configuración vacía
    await plugin.inicializar(nucleus, config)
    comando = {"action": "procesar_bloque", "params": {"bloque_id": "test_block"}}
    with patch.object(plugin.logger, "info", new=MagicMock()) as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        result = await plugin.manejar_comando(comando)
        assert result == {"status": "success", "bloque_id": "test_block"}
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_plugin_comando_invalido(nucleus):
    """Prueba el manejo de comandos inválidos en ExamplePlugin."""
    plugin = ExamplePlugin()
    config = {}  # Configuración vacía
    await plugin.inicializar(nucleus, config)
    comando = {"action": "invalid_action"}
    with patch.object(plugin.logger, "error", new=MagicMock()) as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        result = await plugin.manejar_comando(comando)
        assert result == {"status": "error", "message": "Acción no soportada"}
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_plugin_bloque_procesamiento(nucleus):
    """Prueba el procesamiento de bloques en ExamplePlugin."""
    plugin = ExamplePlugin()
    config = {}  # Configuración vacía
    await plugin.inicializar(nucleus, config)
    with patch.object(plugin.logger, "info", new=MagicMock()) as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        result = await plugin.procesar_bloque("test_block")
        assert result == {"status": "success", "bloque_id": "test_block"}
        assert mock_alerta.called
        assert mock_logger.called
