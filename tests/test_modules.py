import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad


@pytest.mark.asyncio
async def test_modulo_registro_inicializar(nucleus):
    """Prueba la inicialización de ModuloRegistro."""
    registro = ModuloRegistro()
    with patch("corec.blocks.BloqueSimbiotico") as mock_bloque, \
         patch.object(registro.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta, \
         patch("corec.db.init_postgresql") as mock_init_db, \
         patch("aioredis.from_url", new=AsyncMock()) as mock_redis_url:
        nucleus.config["bloques"] = [{"id": "test_block", "canal": 1, "entidades": 1000}]
        await asyncio.wait_for(registro.inicializar(nucleus), timeout=5)
        assert mock_bloque.called
        assert mock_init_db.called
        assert mock_redis_url.called
        assert "test_block" in registro.bloques
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque(nucleus):
    """Prueba el registro de un bloque en ModuloRegistro."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(registro.inicializar(nucleus), timeout=5)
        await asyncio.wait_for(registro.registrar_bloque("new_block", 2, 500), timeout=5)
        assert "new_block" in registro.bloques
        assert registro.bloques["new_block"].canal == 2
        assert len(registro.bloques["new_block"].entidades) == 500
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque_config_invalida(nucleus):
    """Prueba el registro de un bloque con configuración inválida."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        nucleus.config["bloques"] = [{"id": "invalid_block", "canal": -1, "entidades": 500}]
        await asyncio.wait_for(registro.inicializar(nucleus), timeout=5)
        assert "invalid_block" not in registro.bloques
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_inicializar(nucleus):
    """Prueba la inicialización de ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    with patch.object(sincronizacion.logger, "info") as mock_logger:
        await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
        assert sincronizacion.nucleus == nucleus
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades(nucleus):
    """Prueba la redirección de entidades en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], nucleus=nucleus)
    bloque2 = BloqueSimbiotico("block2", 1, entidades[500:], nucleus=nucleus)
    bloque1.fitness = 0.2
    bloque2.fitness = 0.9
    registro.bloques["block1"] = bloque1
    registro.bloques["block2"] = bloque2
    with patch.object(sincronizacion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(sincronizacion.redirigir_entidades("block1", "block2", 200, 1), timeout=5)
        assert len(bloque1.entidades) == 300
        assert len(bloque2.entidades) == 700
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades_error(nucleus):
    """Prueba la redirección de entidades con bloques inexistentes."""
    sincronizacion = ModuloSincronizacion()
    await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
    with patch.object(sincronizacion.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(sincronizacion.redirigir_entidades("block1", "block2", 200, 1), timeout=5)
        assert mock_logger.called
        assert not mock_alerta.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_adaptar_bloque_fusionar(nucleus):
    """Prueba la fusión de bloques en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], nucleus=nucleus)
    bloque2 = BloqueSimbiotico("block2", 1, entidades[500:], nucleus=nucleus)
    bloque1.fitness = 0.1
    bloque2.fitness = 0.6
    registro.bloques["block1"] = bloque1
    registro.bloques["block2"] = bloque2
    with patch.object(sincronizacion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(sincronizacion.adaptar_bloque("block1", carga=0.1), timeout=5)
        assert "block1" not in registro.bloques
        assert "block2" not in registro.bloques
        assert any("fus_" in bid for bid in registro.bloques)
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_ejecucion_inicializar(nucleus):
    """Prueba la inicialización de ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    with patch.object(ejecucion.logger, "info") as mock_logger:
        await asyncio.wait_for(ejecucion.inicializar(nucleus), timeout=5)
        assert ejecucion.nucleus == nucleus
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas(nucleus):
    """Prueba el encolado de tareas en ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    await asyncio.wait_for(ejecucion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, nucleus=nucleus)
    registro.bloques["test_block"] = bloque
    with patch.object(ejecucion, "ejecutar_bloque_task") as mock_task, \
         patch.object(ejecucion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(ejecucion.ejecutar(), timeout=5)
        assert mock_task.delay.called
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas_error(nucleus):
    """Prueba el encolado de tareas con error."""
    ejecucion = ModuloEjecucion()
    await asyncio.wait_for(ejecucion.inicializar(nucleus), timeout=5)
    with patch.object(ejecucion, "ejecutar_bloque_task", side_effect=Exception("Task error")), \
         patch.object(ejecucion.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(ejecucion.ejecutar(), timeout=5)
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_auditoria_inicializar(nucleus):
    """Prueba la inicialización de ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    with patch.object(auditoria.logger, "info") as mock_logger:
        await asyncio.wait_for(auditoria.inicializar(nucleus), timeout=5)
        assert auditoria.nucleus == nucleus
        assert auditoria.detector is not None
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias(nucleus, mock_postgresql):
    """Prueba la detección de anomalías en ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    await asyncio.wait_for(auditoria.inicializar(nucleus), timeout=5)
    with patch("psycopg2.connect", return_value=mock_postgresql), \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        mock_postgresql.cursor.return_value.fetchall.side_effect = [
            [(100, 0.9), (200, 0.1)],  # Datos
            [("block1"), ("block2")]    # IDs
        ]
        mock_postgresql.cursor.return_value.__enter__.return_value.execute.side_effect = None
        await asyncio.wait_for(auditoria.detectar_anomalias(), timeout=5)
        assert mock_postgresql.cursor.called
        assert mock_alerta.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias_error(nucleus, mock_postgresql):
    """Prueba la detección de anomalías con error en PostgreSQL."""
    auditoria = ModuloAuditoria()
    await asyncio.wait_for(auditoria.inicializar(nucleus), timeout=5)
    with patch("psycopg2.connect", return_value=mock_postgresql), \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        mock_postgresql.cursor.side_effect = Exception("Database error")
        await asyncio.wait_for(auditoria.detectar_anomalias(), timeout=5)
        assert mock_postgresql.cursor.called
        assert not mock_alerta.called
    await nucleus.detener()
