import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from corec.nucleus import CoreCNucleus
from pydantic import ValidationError


@pytest.fixture
def mock_config():
    return {
        "db_config": {"host": "localhost"},
        "redis_config": {"host": "localhost", "port": 6379},
        "bloques": [],
        "plugins": {}
    }


@pytest.mark.asyncio
async def test_nucleus_inicializar_exitoso(mock_postgresql, mock_config):
    """Prueba la inicialización exitosa del núcleo."""
    nucleus = CoreCNucleus("config.yml")
    with patch("corec.nucleus.init_postgresql") as mock_init_db, \
            patch("corec.nucleus.aioredis.from_url", new_callable=MagicMock) as mock_redis, \
            patch("corec.nucleus.cargar_config", return_value=mock_config), \
            patch.object(nucleus.logger, "info") as mock_logger, \
            patch("corec.nucleus.ModuloRegistro") as mock_registro, \
            patch("corec.nucleus.ModuloSincronizacion") as mock_sincro, \
            patch("corec.nucleus.ModuloEjecucion") as mock_ejecucion, \
            patch("corec.nucleus.ModuloAuditoria") as mock_auditoria:
        mock_redis.return_value = MagicMock()  # Usamos MagicMock para evitar await en el mock
        mock_registro.return_value.inicializar = MagicMock()
        mock_sincro.return_value.inicializar = MagicMock()
        mock_ejecucion.return_value.inicializar = MagicMock()
        mock_auditoria.return_value.inicializar = MagicMock()
        await nucleus.inicializar()
        assert mock_init_db.called
        assert mock_redis.called
        assert mock_logger.called
        assert len(nucleus.modules) == 4


@pytest.mark.asyncio
async def test_nucleus_inicializar_redis_error(mock_postgresql, mock_config):
    """Prueba la inicialización con un error en Redis."""
    nucleus = CoreCNucleus("config.yml")
    with patch("corec.nucleus.init_postgresql") as mock_init_db, \
            patch("corec.nucleus.aioredis.from_url", side_effect=Exception("Redis Error"), new_callable=MagicMock) as mock_redis, \
            patch("corec.nucleus.cargar_config", return_value=mock_config), \
            patch.object(nucleus.logger, "error") as mock_logger, \
            patch("corec.nucleus.ModuloRegistro") as mock_registro, \
            patch("corec.nucleus.ModuloSincronizacion") as mock_sincro, \
            patch("corec.nucleus.ModuloEjecucion") as mock_ejecucion, \
            patch("corec.nucleus.ModuloAuditoria") as mock_auditoria:
        mock_redis.return_value = MagicMock()
        mock_registro.return_value.inicializar = MagicMock()
        mock_sincro.return_value.inicializar = MagicMock()
        mock_ejecucion.return_value.inicializar = MagicMock()
        mock_auditoria.return_value.inicializar = MagicMock()
        await nucleus.inicializar()
        assert mock_init_db.called
        assert mock_redis.called
        assert mock_logger.called
        assert nucleus.redis_client is None


@pytest.mark.asyncio
async def test_nucleus_inicializar_bloque_exitoso(mock_postgresql, mock_config):
    """Prueba la inicialización con un bloque exitoso."""
    mock_config["bloques"] = [{"id": "block_1", "canal": 1, "entidades": 1}]
    nucleus = CoreCNucleus("config.yml")
    with patch("corec.nucleus.init_postgresql"), \
            patch("corec.nucleus.aioredis.from_url", return_value=MagicMock()), \
            patch("corec.nucleus.cargar_config", return_value=mock_config), \
            patch("corec.nucleus.PluginBlockConfig") as mock_config_class, \
            patch("corec.nucleus.crear_entidad") as mock_entidad, \
            patch("corec.nucleus.BloqueSimbiotico") as mock_bloque, \
            patch("corec.nucleus.ModuloRegistro") as mock_registro, \
            patch("corec.nucleus.ModuloSincronizacion") as mock_sincro, \
            patch("corec.nucleus.ModuloEjecucion") as mock_ejecucion, \
            patch("corec.nucleus.ModuloAuditoria") as mock_auditoria:
        mock_registro.return_value.inicializar = MagicMock()
        mock_registro.return_value.registrar_bloque = AsyncMock()
        mock_sincro.return_value.inicializar = MagicMock()
        mock_ejecucion.return_value.inicializar = MagicMock()
        mock_auditoria.return_value.inicializar = MagicMock()
        await nucleus.inicializar()
        assert mock_registro.return_value.registrar_bloque.called
        assert mock_config_class.called
        assert mock_entidad.called
        assert mock_bloque.called


@pytest.mark.asyncio
async def test_nucleus_inicializar_bloque_config_invalida(mock_postgresql, mock_config):
    """Prueba la inicialización con un bloque de configuración inválida."""
    mock_config["bloques"] = [{"id": "block_1", "canal": 1, "entidades": 1}]
    nucleus = CoreCNucleus("config.yml")
    nucleus.redis_client = AsyncMock()  # Mockeamos redis_client para que publicar_alerta funcione
    with patch("corec.nucleus.init_postgresql"), \
            patch("corec.nucleus.aioredis.from_url", return_value=MagicMock()), \
            patch("corec.nucleus.cargar_config", return_value=mock_config), \
            patch("corec.nucleus.PluginBlockConfig", side_effect=ValidationError("Invalid config", [])) as mock_config_class, \
            patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta, \
            patch("corec.nucleus.ModuloRegistro") as mock_registro, \
            patch("corec.nucleus.ModuloSincronizacion") as mock_sincro, \
            patch("corec.nucleus.ModuloEjecucion") as mock_ejecucion, \
            patch("corec.nucleus.ModuloAuditoria") as mock_auditoria:
        mock_registro.return_value.inicializar = MagicMock()
        mock_registro.return_value.registrar_bloque = AsyncMock()
        mock_sincro.return_value.inicializar = MagicMock()
        mock_ejecucion.return_value.inicializar = MagicMock()
        mock_auditoria.return_value.inicializar = MagicMock()
        await nucleus.inicializar()
        assert mock_alerta.called
        assert mock_config_class.called


@pytest.mark.asyncio
async def test_nucleus_registrar_plugin_exitoso(mock_config):
    """Prueba el registro exitoso de un plugin."""
    mock_config["plugins"] = {"test_plugin": {"bloque": {"id": "block_1", "canal": 1, "entidades": 1}}}
    nucleus = CoreCNucleus("config.yml")
    plugin = MagicMock()
    with patch("corec.nucleus.cargar_config", return_value=mock_config), \
            patch("corec.nucleus.PluginBlockConfig") as mock_config_class, \
            patch("corec.nucleus.crear_entidad") as mock_entidad, \
            patch("corec.nucleus.BloqueSimbiotico") as mock_bloque, \
            patch.object(nucleus.logger, "info") as mock_logger:
        nucleus.registrar_plugin("test_plugin", plugin)
        assert nucleus.plugins["test_plugin"] == plugin
        assert mock_logger.called
        assert mock_config_class.called
        assert mock_entidad.called
        assert mock_bloque.called


@pytest.mark.asyncio
async def test_nucleus_registrar_plugin_config_invalida(mock_config):
    """Prueba el registro de un plugin con configuración inválida."""
    mock_config["plugins"] = {"test_plugin": {"bloque": {"id": "block_1", "canal": 1, "entidades": 1}}}
    nucleus = CoreCNucleus("config.yml")
    plugin = MagicMock()
    with patch("corec.nucleus.cargar_config", return_value=mock_config), \
            patch("corec.nucleus.PluginBlockConfig", side_effect=ValidationError("Invalid config", [])) as mock_config_class, \
            patch.object(nucleus.logger, "error") as mock_logger:
        nucleus.registrar_plugin("test_plugin", plugin)
        assert nucleus.plugins["test_plugin"] == plugin
        assert mock_logger.called
        assert mock_config_class.called


@pytest.mark.asyncio
async def test_nucleus_ejecutar_plugin_exitoso():
    """Prueba la ejecución exitosa de un plugin."""
    nucleus = CoreCNucleus("config.yml")
    plugin = MagicMock(manejar_comando=AsyncMock(return_value={"status": "success"}))
    nucleus.plugins["test_plugin"] = plugin
    with patch("corec.nucleus.cargar_config", return_value={}), \
            patch("corec.nucleus.PluginCommand") as mock_command, \
            patch.object(nucleus.logger, "info") as mock_logger:
        result = await nucleus.ejecutar_plugin("test_plugin", {"action": "test"})
        assert result == {"status": "success"}
        assert mock_logger.called
        assert mock_command.called


@pytest.mark.asyncio
async def test_nucleus_ejecutar_plugin_no_encontrado():
    """Prueba la ejecución de un plugin no encontrado."""
    nucleus = CoreCNucleus("config.yml")
    with patch("corec.nucleus.cargar_config", return_value={}):
        with pytest.raises(ValueError, match="Plugin 'test_plugin' no encontrado"):
            await nucleus.ejecutar_plugin("test_plugin", {"action": "test"})


@pytest.mark.asyncio
async def test_nucleus_ejecutar_plugin_comando_invalido():
    """Prueba la ejecución de un plugin con un comando inválido."""
    nucleus = CoreCNucleus("config.yml")
    plugin = MagicMock()
    nucleus.plugins["test_plugin"] = plugin
    with patch("corec.nucleus.cargar_config", return_value={}), \
            patch("corec.nucleus.PluginCommand", side_effect=ValidationError("Invalid command", [])) as mock_command, \
            patch.object(nucleus.logger, "error") as mock_logger:
        result = await nucleus.ejecutar_plugin("test_plugin", {"action": "test"})
        assert result["status"] == "error"
        assert mock_logger.called
        assert mock_command.called


@pytest.mark.asyncio
async def test_nucleus_publicar_alerta_exitoso():
    """Prueba la publicación exitosa de una alerta."""
    nucleus = CoreCNucleus("config.yml")
    nucleus.redis_client = AsyncMock()
    with patch("corec.nucleus.cargar_config", return_value={}), \
            patch.object(nucleus.logger, "warning") as mock_logger:
        await nucleus.publicar_alerta({"tipo": "test"})
        assert nucleus.redis_client.xadd.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_nucleus_detener_exitoso():
    """Prueba la detención exitosa del núcleo."""
    nucleus = CoreCNucleus("config.yml")
    nucleus.redis_client = AsyncMock()
    nucleus.modules = {"modulo1": MagicMock(detener=AsyncMock())}
    with patch("corec.nucleus.cargar_config", return_value={}), \
            patch.object(nucleus.logger, "info") as mock_logger:
        await nucleus.detener()
        assert nucleus.redis_client.close.called
        assert mock_logger.called
