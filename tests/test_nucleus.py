import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus


@pytest.mark.asyncio
async def test_nucleus_inicializar(nucleus, mock_redis):
    """Prueba la inicialización de CoreCNucleus."""
    with patch("corec.db.init_postgresql") as mock_init_db:
        await nucleus.inicializar()
        assert mock_init_db.called
        assert nucleus.redis_client == mock_redis
        assert "registro" in nucleus.modules
        assert "sincronizacion" in nucleus.modules
        assert "ejecucion" in nucleus.modules
        assert "auditoria" in nucleus.modules


@pytest.mark.asyncio
async def test_nucleus_registrar_plugin(nucleus):
    """Prueba el registro de un plugin con bloque simbiótico."""
    plugin = AsyncMock()
    plugin.manejar_comando = AsyncMock()
    nucleus.registrar_plugin("test_plugin", plugin)
    assert "test_plugin" in nucleus.plugins
    assert "test_plugin" in nucleus.bloques_plugins
    bloque = nucleus.bloques_plugins["test_plugin"]
    assert bloque.id == "test_plugin_block"
    assert bloque.canal == 4
    assert len(bloque.entidades) == 500
    assert nucleus.logger.info.called


@pytest.mark.asyncio
async def test_nucleus_registrar_plugin_config_invalida(nucleus):
    """Prueba el registro de un plugin con configuración inválida."""
    nucleus.config["plugins"]["test_plugin"]["bloque"]["entidades"] = -1  # Configuración inválida
    plugin = AsyncMock()
    nucleus.registrar_plugin("test_plugin", plugin)
    assert "test_plugin" in nucleus.plugins
    assert "test_plugin" not in nucleus.bloques_plugins  # No se crea el bloque
    assert nucleus.logger.error.called


@pytest.mark.asyncio
async def test_nucleus_ejecutar_plugin(nucleus):
    """Prueba la ejecución de un comando en un plugin."""
    plugin = AsyncMock()
    plugin.manejar_comando = AsyncMock(return_value={"status": "success"})
    nucleus.registrar_plugin("test_plugin", plugin)
    comando = {"action": "test", "params": {"key": "value"}}
    resultado = await nucleus.ejecutar_plugin("test_plugin", comando)
    assert resultado == {"status": "success"}
    assert plugin.manejar_comando.called


@pytest.mark.asyncio
async def test_nucleus_ejecutar_plugin_comando_invalido(nucleus):
    """Prueba la ejecución de un comando inválido en un plugin."""
    plugin = AsyncMock()
    plugin.manejar_comando = AsyncMock()
    nucleus.registrar_plugin("test_plugin", plugin)
    comando = {}  # Falta 'action'
    resultado = await nucleus.ejecutar_plugin("test_plugin", comando)
    assert resultado["status"] == "error"
    assert "Comando inválido" in resultado["message"]
    assert not plugin.manejar_comando.called


@pytest.mark.asyncio
async def test_nucleus_publicar_alerta(nucleus, mock_redis):
    """Prueba la publicación de una alerta."""
    alerta = {"tipo": "test_alerta", "mensaje": "Test"}
    await nucleus.publicar_alerta(alerta)
    assert mock_redis.xadd.called
    assert nucleus.logger.warning.called


@pytest.mark.asyncio
async def test_nucleus_coordinar_bloques(nucleus, mock_redis):
    """Prueba la coordinación de bloques simbióticos."""
    plugin = AsyncMock()
    plugin.manejar_comando = AsyncMock()
    nucleus.registrar_plugin("test_plugin", plugin)
    with patch("corec.blocks.BloqueSimbiotico.procesar", new=AsyncMock()) as mock_procesar:
        await nucleus.coordinar_bloques()
        assert mock_procesar.called
        assert nucleus.logger.debug.called
