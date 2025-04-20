import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.mark.asyncio
async def test_nucleus_inicializar(nucleus, mock_redis):
    """Prueba la inicialización de CoreCNucleus."""
    with patch("corec.db.init_postgresql") as mock_init_db, \
         patch("corec.nucleus.logging.getLogger") as mock_logging, \
         patch("aioredis.from_url", new=AsyncMock()) as mock_redis_url:
        mock_logging.return_value.info = MagicMock()
        await asyncio.wait_for(nucleus.inicializar(), timeout=5)
        assert mock_init_db.called
        assert "registro" in nucleus.modules
        assert "sincronizacion" in nucleus.modules
        assert "ejecucion" in nucleus.modules
        assert "auditoria" in nucleus.modules
        assert mock_logging.return_value.info.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_nucleus_inicializar_modulo_no_encontrado(nucleus, mock_redis):
    """Prueba la inicialización con un módulo faltante."""
    with patch("corec.db.init_postgresql"), \
         patch("pathlib.Path.glob") as mock_glob, \
         patch("corec.nucleus.logging.getLogger") as mock_logging:
        mock_glob.return_value = []
        mock_logging.return_value.info = MagicMock()
        await asyncio.wait_for(nucleus.inicializar(), timeout=5)
        assert "registro" in nucleus.modules
        assert mock_logging.return_value.info.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_nucleus_registrar_plugin(nucleus):
    """Prueba el registro de un plugin con bloque simbiótico."""
    plugin = AsyncMock()
    plugin.manejar_comando = AsyncMock()
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        mock_logging.return_value.info = MagicMock()
        nucleus.config["plugins"] = {
            "example_plugin": {
                "enabled": True,
                "bloque": {"id": "example_plugin_block", "canal": 4, "entidades": 500}
            }
        }
        nucleus.registrar_plugin("example_plugin", plugin)
        assert "example_plugin" in nucleus.plugins
        assert "example_plugin" in nucleus.bloques_plugins
        bloque = nucleus.bloques_plugins["example_plugin"]
        assert bloque.id == "example_plugin_block"
        assert bloque.canal == 4
        assert len(bloque.entidades) == 500
        assert mock_logging.return_value.info.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_nucleus_registrar_plugin_config_invalida(nucleus):
    """Prueba el registro de un plugin con configuración inválida."""
    plugin = AsyncMock()
    plugin.manejar_comando = AsyncMock()
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        mock_logging.return_value.error = MagicMock()
        nucleus.config["plugins"] = {
            "example_plugin": {
                "enabled": True,
                "bloque": {"id": "example_plugin_block", "canal": -1, "entidades": 500}
            }
        }
        nucleus.registrar_plugin("example_plugin", plugin)
        assert "example_plugin" in nucleus.plugins
        assert "example_plugin" not in nucleus.bloques_plugins
        assert mock_logging.return_value.error.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_nucleus_ejecutar_plugin(nucleus):
    """Prueba la ejecución de un comando en un plugin."""
    plugin = AsyncMock()
    plugin.manejar_comando = AsyncMock(return_value={"status": "success"})
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        mock_logging.return_value.info = MagicMock()
        nucleus.registrar_plugin("example_plugin", plugin)
        comando = {"action": "test", "params": {"key": "value"}}
        resultado = await asyncio.wait_for(nucleus.ejecutar_plugin("example_plugin", comando), timeout=5)
        assert resultado == {"status": "success"}
        assert plugin.manejar_comando.called
        assert mock_logging.return_value.info.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_nucleus_ejecutar_plugin_comando_invalido(nucleus):
    """Prueba la ejecución de un comando inválido en un plugin."""
    plugin = AsyncMock()
    plugin.manejar_comando = AsyncMock()
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        mock_logging.return_value.error = MagicMock()
        nucleus.registrar_plugin("example_plugin", plugin)
        comando = {}
        resultado = await asyncio.wait_for(nucleus.ejecutar_plugin("example_plugin", comando), timeout=5)
        assert resultado["status"] == "error"
        assert "Comando inválido" in resultado["message"]
        assert not plugin.manejar_comando.called
        assert mock_logging.return_value.error.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_nucleus_ejecutar_plugin_no_existe(nucleus):
    """Prueba la ejecución de un comando en un plugin inexistente."""
    comando = {"action": "test", "params": {"key": "value"}}
    with pytest.raises(ValueError, match="Plugin 'example_plugin' no encontrado"):
        await asyncio.wait_for(nucleus.ejecutar_plugin("example_plugin", comando), timeout=5)
    await nucleus.detener()


@pytest.mark.asyncio
async def test_nucleus_publicar_alerta(nucleus, mock_redis):
    """Prueba la publicación de una alerta."""
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        mock_logging.return_value.warning = MagicMock()
        alerta = {"tipo": "test_alerta", "mensaje": "Test"}
        nucleus.redis_client = mock_redis
        await asyncio.wait_for(nucleus.publicar_alerta(alerta), timeout=5)
        assert mock_redis.xadd.called
        assert mock_logging.return_value.warning.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_nucleus_publicar_alerta_error_redis(nucleus, mock_redis):
    """Prueba la publicación de una alerta con error en Redis."""
    mock_redis.xadd.side_effect = Exception("Redis error")
    with patch("corec.nucleus.logging.getLogger") as mock_logging:
        mock_logging.return_value.error = MagicMock()
        alerta = {"tipo": "test_alerta", "mensaje": "Test"}
        nucleus.redis_client = mock_redis
        await asyncio.wait_for(nucleus.publicar_alerta(alerta), timeout=5)
        assert mock_redis.xadd.called
        assert mock_logging.return_value.error.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_nucleus_coordinar_bloques(nucleus, mock_redis):
    """Prueba la coordinación de bloques simbióticos."""
    plugin = AsyncMock()
    plugin.manejar_comando = AsyncMock()
    with patch("corec.blocks.BloqueSimbiotico.procesar", new=AsyncMock()) as mock_procesar, \
         patch("corec.nucleus.logging.getLogger") as mock_logging:
        mock_logging.return_value.debug = MagicMock()
        nucleus.config["plugins"] = {
            "example_plugin": {
                "enabled": True,
                "bloque": {"id": "example_plugin_block", "canal": 4, "entidades": 500}
            }
        }
        nucleus.registrar_plugin("example_plugin", plugin)
        await asyncio.wait_for(nucleus.coordinar_bloques(), timeout=5)
        assert mock_procesar.called
        assert mock_logging.return_value.debug.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_nucleus_coordinar_bloques_error(nucleus, mock_redis):
    """Prueba la coordinación de bloques con error."""
    plugin = AsyncMock()
    plugin.manejar_comando = AsyncMock()
    with patch("corec.blocks.BloqueSimbiotico.procesar", side_effect=Exception("Process error")), \
         patch("corec.nucleus.logging.getLogger") as mock_logging, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        mock_logging.return_value.error = MagicMock()
        nucleus.config["plugins"] = {
            "example_plugin": {
                "enabled": True,
                "bloque": {"id": "example_plugin_block", "canal": 4, "entidades": 500}
            }
        }
        nucleus.registrar_plugin("example_plugin", plugin)
        await asyncio.wait_for(nucleus.coordinar_bloques(), timeout=5)
        assert mock_alerta.called
        assert mock_logging.return_value.error.called
    await nucleus.detener()
