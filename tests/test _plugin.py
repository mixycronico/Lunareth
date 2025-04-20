import pytest
from unittest.mock import AsyncMock, patch
from plugins.example_plugin.main import inicializar


@pytest.mark.asyncio
async def test_plugin_inicializar(nucleus):
    """Prueba la inicialización del plugin example_plugin."""
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        plugin = await inicializar(nucleus, {})
        assert "example_plugin" in nucleus.plugins
        assert "example_plugin" in nucleus.bloques_plugins
        bloque = nucleus.bloques_plugins["example_plugin"]
        assert bloque.id == "example_plugin_block"
        assert bloque.canal == 4
        assert len(bloque.entidades) == 500
        assert mock_logging.return_value.info.called


@pytest.mark.asyncio
async def test_plugin_manejar_comando(nucleus):
    """Prueba el manejo de comandos por example_plugin."""
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        plugin = await inicializar(nucleus, {})
        comando = {"action": "test_action", "params": {"key": "value"}}
        resultado = await nucleus.ejecutar_plugin("example_plugin", comando)
        assert resultado == {"status": "success", "action": "test_action"}
        assert mock_logging.return_value.info.called


@pytest.mark.asyncio
async def test_plugin_comando_invalido(nucleus):
    """Prueba el manejo de un comando inválido por example_plugin."""
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        plugin = await inicializar(nucleus, {})
        comando = {}  # Falta 'action'
        resultado = await nucleus.ejecutar_plugin("example_plugin", comando)
        assert resultado["status"] == "error"
        assert "Comando inválido" in resultado["message"]
        assert mock_logging.return_value.error.called


@pytest.mark.asyncio
async def test_plugin_bloque_procesamiento(nucleus):
    """Prueba el procesamiento del bloque simbiótico de example_plugin."""
    with patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        plugin = await inicializar(nucleus, {})
        bloque = nucleus.bloques_plugins["example_plugin"]
        resultado = await bloque.procesar(carga=0.5)
        assert resultado["bloque_id"] == "example_plugin_block"
        assert len(resultado["mensajes"]) == 500
        assert resultado["fitness"] >= 0.0
        assert mock_alerta.called
