import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from corec.modules.cognitivo import ModuloCognitivo
from corec.modules.autosanacion import ModuloAutosanacion
from corec.entities import Entidad
from corec.blocks import BloqueSimbiotico
from corec.nucleus import CoreCNucleus

@pytest.mark.asyncio
async def test_modulo_registro_inicializar(nucleus):
    """Prueba la inicialización de ModuloRegistro."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "info") as mock_logger:
        await registro.inicializar(nucleus)
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque(nucleus):
    """Prueba el registro de un bloque en ModuloRegistro."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await registro.inicializar(nucleus)
        await registro.registrar_bloque("new_block", 2, 500, max_size_mb=10.0)
        assert "new_block" in registro.bloques
        assert mock_alerta.called
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque_config_invalida(nucleus):
    """Prueba el registro de un bloque con configuración inválida en ModuloRegistro."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await registro.inicializar(nucleus)
        with pytest.raises(ValueError):
            await registro.registrar_bloque(None, -1, 0)
        assert mock_alerta.called
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_registro_detener(nucleus):
    """Prueba la detención de ModuloRegistro."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "info") as mock_logger:
        await registro.inicializar(nucleus)
        await registro.detener()
        assert mock_logger.called_with_call("Módulo Registro detenido")

@pytest.mark.asyncio
async def test_modulo_sincronizacion_inicializar(nucleus):
    """Prueba la inicialización de ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    with patch.object(sincronizacion.logger, "info") as mock_logger:
        await sincronizacion.inicializar(nucleus)
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades(nucleus):
    """Prueba la redirección de entidades en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await sincronizacion.inicializar(nucleus)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    entidades = [Entidad(f"m{i}", 1, lambda carga: {"valor": 0.7}) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    bloque2 = BloqueSimbiotico("block2", 2, entidades[500:], 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    with patch.object(sincronizacion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await sincronizacion.redirigir_entidades(bloque1, bloque2, 0.1, canal=2)
        assert len(bloque1.entidades) == 450
        assert len(bloque2.entidades) == 550
        assert mock_alerta.called
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades_error(nucleus):
    """Prueba la redirección de entidades con un error en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await sincronizacion.inicializar(nucleus)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    entidades = [Entidad(f"m{i}", 1, lambda carga: {"valor": 0.7}) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    bloque2 = BloqueSimbiotico("block2", 2, entidades[500:], 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    with patch.object(sincronizacion.logger, "error") as mock_logger, \
         patch("random.uniform", side_effect=Exception("Error")):
        with pytest.raises(Exception):
            await sincronizacion.redirigir_entidades(bloque1, bloque2, 0.1, canal=2)
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_sincronizacion_adaptar_bloque_fusionar(nucleus):
    """Prueba la fusión de bloques en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await sincronizacion.inicializar(nucleus)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    entidades = [Entidad(f"m{i}", 1, lambda carga: {"valor": 0.7}) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    bloque2 = BloqueSimbiotico("block2", 2, entidades[500:], 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    with patch.object(sincronizacion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await sincronizacion.adaptar_bloque(bloque1, bloque2)
        assert len(bloque1.entidades) == 0
        assert len(bloque2.entidades) == 1000
        assert mock_alerta.called
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_sincronizacion_detener(nucleus):
    """Prueba la detención de ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    with patch.object(sincronizacion.logger, "info") as mock_logger:
        await sincronizacion.inicializar(nucleus)
        await sincronizacion.detener()
        assert mock_logger.called_with_call("Módulo Sincronización detenido")

@pytest.mark.asyncio
async def test_modulo_ejecucion_inicializar(nucleus):
    """Prueba la inicialización de ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    with patch.object(ejecucion.logger, "info") as mock_logger:
        await ejecucion.inicializar(nucleus)
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas(nucleus):
    """Prueba el encolado de tareas en ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    await ejecucion.inicializar(nucleus)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    entidades = [Entidad(f"m{i}", 1, lambda carga: {"valor": 0.7}) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    with patch.object(ejecucion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await ejecucion.encolar_bloque(bloque)
        assert len(bloque.mensajes) == 100
        assert mock_alerta.called
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas_error(nucleus):
    """Prueba el encolado de tareas con un error en ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    await ejecucion.inicializar(nucleus)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    entidades = [Entidad(f"m{i}", 1, lambda carga: {"valor": 0.7}) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus, quantization_step=0.05, max_errores=0.1)
    with patch.object(ejecucion.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta, \
         patch("random.uniform", side_effect=Exception("Error")):
        with pytest.raises(Exception):
            await ejecucion.encolar_bloque(bloque)
        assert mock_alerta.called
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_ejecucion_detener(nucleus):
    """Prueba la detención de ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    with patch.object(ejecucion.logger, "info") as mock_logger:
        await ejecucion.inicializar(nucleus)
        await ejecucion.detener()
        assert mock_logger.called_with_call("Módulo Ejecución detenido")

@pytest.mark.asyncio
async def test_modulo_auditoria_inicializar(nucleus):
    """Prueba la inicialización de ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    with patch.object(auditoria.logger, "info") as mock_logger:
        await auditoria.inicializar(nucleus)
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias(nucleus):
    """Prueba la detección de anomalías en ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    await auditoria.inicializar(nucleus)
    registro = ModuloRegistro()
    registro.bloques = {"block1": {"fitness": -1.0, "num_entidades": 10}}
    nucleus.modules["registro"] = registro
    with patch.object(auditoria.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await auditoria.detectar_anomalias()
        assert mock_alerta.called
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias_error(nucleus):
    """Prueba la detección de anomalías con un error en ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    await auditoria.inicializar(nucleus)
    registro = ModuloRegistro()
    registro.bloques = {"block1": {"fitness": -1.0, "num_entidades": 10}}
    nucleus.modules["registro"] = registro
    with patch.object(auditoria.logger, "error") as mock_logger, \
         patch("random.random", side_effect=Exception("Error")):
        with pytest.raises(Exception):
            await auditoria.detectar_anomalias()
        assert mock_logger.called

@pytest.mark.asyncio
async def test_modulo_auditoria_detener(nucleus):
    """Prueba la detención de ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    with patch.object(auditoria.logger, "info") as mock_logger:
        await auditoria.inicializar(nucleus)
        await auditoria.detener()
        assert mock_logger.called_with_call("Módulo Auditoría detenido")

@pytest.mark.asyncio
async def test_modulo_cognitivo_inicializar(nucleus):
    """Prueba la inicialización de ModuloCognitivo."""
    cognitivo = ModuloCognitivo()
    with patch.object(cognitivo.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await cognitivo.inicializar(nucleus)
        assert mock_logger.called
        assert mock_alerta.called

@pytest.mark.asyncio
async def test_modulo_cognitivo_guardar_estado(nucleus, mock_db_pool):
    """Prueba el guardado de estado limitado en ModuloCognitivo."""
    cognitivo = ModuloCognitivo()
    await cognitivo.inicializar(nucleus)
    cognitivo.percepciones = [{"tipo": "test", "valor": 0.5, "timestamp": 12345}] * 200  # Más de 100 percepciones
    cognitivo.decisiones = [{"opcion": "test", "confianza": 0.5, "timestamp": 12345}] * 200  # Más de 100 decisiones
    nucleus.db_pool = mock_db_pool
    with patch.object(cognitivo.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await cognitivo.guardar_estado()
        assert mock_logger.called
        assert mock_alerta.called
        assert len(json.loads(mock_db_pool.acquire.return_value.__aenter__.return_value.execute.call_args[1][4])) == 100  # Solo 100 percepciones
        assert len(json.loads(mock_db_pool.acquire.return_value.__aenter__.return_value.execute.call_args[1][5])) == 100  # Solo 100 decisiones

@pytest.mark.asyncio
async def test_modulo_autosanacion_reconectar(nucleus, mock_db_pool, mock_redis):
    """Prueba la reconexión automática en ModuloAutosanacion."""
    autosanacion = ModuloAutosanacion()
    await autosanacion.inicializar(nucleus, nucleus.config.autosanacion_config)
    nucleus.db_pool = None
    nucleus.redis_client = None
    with patch("corec.utils.db_utils.init_postgresql", AsyncMock(return_value=mock_db_pool)) as mock_init_postgresql, \
         patch("corec.utils.db_utils.init_redis", AsyncMock(return_value=mock_redis)) as mock_init_redis, \
         patch.object(nucleus, "publicar_alerta", AsyncMock()) as mock_alerta:
        await autosanacion.verificar_estado()
        assert mock_init_postgresql.called
        assert mock_init_redis.called
        assert mock_alerta.called
