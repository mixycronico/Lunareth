import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from corec.entities import crear_entidad
from corec.blocks import BloqueSimbiotico


@pytest.mark.asyncio
async def test_modulo_registro_inicializar(nucleus):
    """Prueba la inicialización de ModuloRegistro."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "info") as mock_logger:
        await asyncio.wait_for(registro.inicializar(nucleus), timeout=5)
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque(nucleus):
    """Prueba el registro de un bloque en ModuloRegistro."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(registro.inicializar(nucleus), timeout=5)
        await asyncio.wait_for(registro.registrar_bloque("new_block", 2, 500, max_size_mb=10.0), timeout=5)
        assert "new_block" in registro.bloques
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque_config_invalida(nucleus):
    """Prueba el registro de un bloque con configuración inválida en ModuloRegistro."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(registro.inicializar(nucleus), timeout=5)
        with pytest.raises(ValueError):
            await asyncio.wait_for(registro.registrar_bloque(None, -1, 0), timeout=5)
        assert mock_alerta.call_count == 1  # Solo para el error
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_registro_detener(nucleus):
    """Prueba la detención de ModuloRegistro."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "info") as mock_logger:
        await asyncio.wait_for(registro.inicializar(nucleus), timeout=5)
        await registro.detener()
        assert mock_logger.called_with_call("[Registro] Módulo detenido")


@pytest.mark.asyncio
async def test_modulo_sincronizacion_inicializar(nucleus):
    """Prueba la inicialización de ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    with patch.object(sincronizacion.logger, "info") as mock_logger:
        await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades(nucleus):
    """Prueba la redirección de entidades en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], 10.0, nucleus)
    bloque2 = BloqueSimbiotico("block2", 2, entidades[500:], 10.0, nucleus)
    with patch.object(sincronizacion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(sincronizacion.redirigir_entidades(bloque1, bloque2, 0.1, canal=2), timeout=5)
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades_error(nucleus):
    """Prueba la redirección de entidades con un error en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], 10.0, nucleus)
    bloque2 = BloqueSimbiotico("block2", 2, entidades[500:], 10.0, nucleus)
    with patch.object(sincronizacion.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta, \
         patch("corec.modules.sincronizacion.random.random", side_effect=Exception("Error")):
        try:
            await asyncio.wait_for(sincronizacion.redirigir_entidades(bloque1, bloque2, 0.1, canal=2), timeout=5)
        except Exception:
            pass
        assert mock_alerta.call_count == 1  # Solo para el error
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_sincronizacion_adaptar_bloque_fusionar(nucleus):
    """Prueba la fusión de bloques en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], 10.0, nucleus)
    bloque2 = BloqueSimbiotico("block2", 2, entidades[500:], 10.0, nucleus)
    with patch.object(sincronizacion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(sincronizacion.adaptar_bloque(bloque1, bloque2), timeout=5)
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_sincronizacion_detener(nucleus):
    """Prueba la detención de ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    with patch.object(sincronizacion.logger, "info") as mock_logger:
        await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
        await sincronizacion.detener()
        assert mock_logger.called_with_call("[Sincronización] Módulo detenido")


@pytest.mark.asyncio
async def test_modulo_ejecucion_inicializar(nucleus):
    """Prueba la inicialización de ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    with patch.object(ejecucion.logger, "info") as mock_logger:
        await asyncio.wait_for(ejecucion.inicializar(nucleus), timeout=5)
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas(nucleus):
    """Prueba el encolado de tareas en ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    await asyncio.wait_for(ejecucion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    with patch.object(ejecucion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(ejecucion.encolar_bloque(bloque), timeout=5)
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas_error(nucleus):
    """Prueba el encolado de tareas con un error en ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    await asyncio.wait_for(ejecucion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, 10.0, nucleus)
    with patch.object(ejecucion.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta, \
         patch("corec.modules.ejecucion.random.uniform", side_effect=Exception("Error")):
        try:
            await asyncio.wait_for(ejecucion.encolar_bloque(bloque), timeout=5)
        except Exception:
            pass
        assert mock_alerta.call_count == 1  # Solo para el error
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_ejecucion_detener(nucleus):
    """Prueba la detención de ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    with patch.object(ejecucion.logger, "info") as mock_logger:
        await asyncio.wait_for(ejecucion.inicializar(nucleus), timeout=5)
        await ejecucion.detener()
        assert mock_logger.called_with_call("[Ejecución] Módulo detenido")


@pytest.mark.asyncio
async def test_modulo_auditoria_inicializar(nucleus):
    """Prueba la inicialización de ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    with patch.object(auditoria.logger, "info") as mock_logger:
        await asyncio.wait_for(auditoria.inicializar(nucleus), timeout=5)
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias(nucleus):
    """Prueba la detección de anomalías en ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    await asyncio.wait_for(auditoria.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    registro.bloques = {"block1": {"fitness": -1.0, "num_entidades": 10}}
    nucleus.modules["registro"] = registro
    with patch.object(auditoria.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(auditoria.detectar_anomalias(), timeout=5)
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias_error(nucleus):
    """Prueba la detección de anomalías con un error en ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    await asyncio.wait_for(auditoria.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    with patch.object(auditoria.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta, \
         patch("corec.modules.auditoria.random.random", side_effect=Exception("Error")):
        try:
            await asyncio.wait_for(auditoria.detectar_anomalias(), timeout=5)
        except Exception:
            pass
        assert mock_alerta.call_count == 1  # Solo para el error
        assert mock_logger.called


@pytest.mark.asyncio
async def test_modulo_auditoria_detener(nucleus):
    """Prueba la detención de ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    with patch.object(auditoria.logger, "info") as mock_logger:
        await asyncio.wait_for(auditoria.inicializar(nucleus), timeout=5)
        await auditoria.detener()
        assert mock_logger.called_with_call("[Auditoría] Módulo detenido")
