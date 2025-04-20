import pytest
import asyncio
from unittest.mock import AsyncMock
from corec.nucleus import CoreCNucleus


@pytest.mark.asyncio
async def test_plugin_inicializar(nucleus):
    """Prueba la inicialización de un plugin genérico."""
    class TestPlugin:
        async def inicializar(self, nucleus, config):
            self.nucleus = nucleus
            self.config = config
            nucleus.registrar_plugin("test_plugin", self)

        async def manejar_comando(self, comando):
            return {"status": "success", "action": comando["action"]}

    plugin = TestPlugin()
    await plugin.inicializar(nucleus, {})
    assert "test_plugin" in nucleus.plugins
    assert "test_plugin" in nucleus.bloques_plugins
    bloque = nucleus.bloques_plugins["test_plugin"]
    assert bloque.id == "test_plugin_block"
    assert bloque.canal == 4
    assert len(bloque.entidades) == 500
    assert nucleus.logger.info.called


@pytest.mark.asyncio
async def test_plugin_manejar_comando(nucleus):
    """Prueba el manejo de comandos por un plugin genérico."""
    class TestPlugin:
        async def inicializar(self, nucleus, config):
            self.nucleus = nucleus
            self.config = config
            nucleus.registrar_plugin("test_plugin", self)

        async def manejar_comando(self, comando):
            return {"status": "success", "action": comando["action"]}

    plugin = TestPlugin()
    await plugin.inicializar(nucleus, {})
    comando = {"action": "test_action", "params": {"key": "value"}}
    resultado = await nucleus.ejecutar_plugin("test_plugin", comando)
    assert resultado == {"status": "success", "action": "test_action"}
    assert nucleus.logger.info.called


@pytest.mark.asyncio
async def test_plugin_comando_invalido(nucleus):
    """Prueba el manejo de un comando inválido por un plugin."""
    class TestPlugin:
        async def inicializar(self, nucleus, config):
            self.nucleus = nucleus
            self.config = config
            nucleus.registrar_plugin("test_plugin", self)

        async def manejar_comando(self, comando):
            return {"status": "success", "action": comando["action"]}

    plugin = TestPlugin()
    await plugin.inicializar(nucleus, {})
    comando = {}  # Falta 'action'
    resultado = await nucleus.ejecutar_plugin("test_plugin", comando)
    assert resultado["status"] == "error"
    assert "Comando inválido" in resultado["message"]
    assert nucleus.logger.error.called
