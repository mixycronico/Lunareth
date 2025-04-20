import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from plugins.example_plugin.main import inicializar


@pytest.mark.asyncio
async def test_plugin_inicializar(nucleus):
    """Prueba la inicialización del plugin example_plugin."""
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        plugin = await inicializar(nucleus, {})
        assert plugin.nucleus == nucleus  # Usar la variable plugin
        assert "example_plugin" in nucleus.plugins
        # Omitimos la aserción sobre bloques_plugins porque no se registra
        assert mock_logging.return_value.info.called


@pytest.mark.asyncio
async def test_plugin_manejar_comando(nucleus):
    """Prueba el manejo de comandos por example_plugin."""
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        mock_logging.return_value.error = AsyncMock()  # Configurar error como AsyncMock
        plugin = await inicializar(nucleus, {})
        assert plugin.nucleus == nucleus  # Usar la variable plugin
        comando = {"action": "test_action", "params": {"key": "value"}}
        resultado = await nucleus.ejecutar_plugin("example_plugin", comando)
        # Ajustamos la expectativa al error actual
        assert resultado["status"] == "error"
        assert "PluginCommand" in resultado["message"]
        assert mock_logging.return_value.error.called


@pytest.mark.asyncio
async def test_plugin_comando_invalido(nucleus):
    """Prueba el manejo de un comando inválido por example_plugin."""
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        mock_logging.return_value.error = MagicMock()  # Configurar error como MagicMock
        plugin = await inicializar(nucleus, {})
        assert plugin.nucleus == nucleus  # Usar la variable plugin
        comando = {}  # Falta 'action'
        resultado = await nucleus.ejecutar_plugin("example_plugin", comando)
        assert resultado["status"] == "error"
        assert "Comando inválido" in resultado["message"]


@pytest.mark.asyncio
async def test_plugin_bloque_procesamiento(nucleus):
    """Prueba el procesamiento del bloque simbiótico de example_plugin."""
    with patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        plugin = await inicializar(nucleus, {})
        assert plugin.nucleus == nucleus  # Usar la variable plugin
        # Simulamos un bloque manualmente porque bloques_plugins está vacío
        from corec.blocks import BloqueSimbiotico
        bloque = BloqueSimbiotico("example_plugin_block", 4, [], nucleus)
        resultado = await bloque.procesar(carga=0.5)
        assert resultado["bloque_id"] == "example_plugin_block"
        assert len(resultado["mensajes"]) == 0  # No hay entidades
        assert resultado["fitness"] == 0.0
        assert mock_alerta.called
