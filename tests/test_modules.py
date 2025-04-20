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
    async with asyncio.timeout(5):  # Timeout de 5 segundos
        registro = ModuloRegistro()
        with patch("corec.blocks.BloqueSimbiotico") as mock_bloque, patch.object(registro.logger, "info") as mock_logger:
            await registro.inicializar(nucleus)
            assert mock_bloque.called
            assert "test_block" in registro.bloques
            assert nucleus.publicar_alerta.called
            assert mock_logger.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque(nucleus):
    """Prueba el registro de un bloque en ModuloRegistro."""
    async with asyncio.timeout(5):
        registro = ModuloRegistro()
        with patch.object(registro.logger, "info") as mock_logger:
            await registro.inicializar(nucleus)
            await registro.registrar_bloque("new_block", 2, 500)
            assert "new_block" in registro.bloques
            assert registro.bloques["new_block"].canal == 2
            assert len(registro.bloques["new_block"].entidades) == 500
            assert nucleus.publicar_alerta.called
            assert mock_logger.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque_config_invalida(nucleus):
    """Prueba el registro de un bloque con configuración inválida."""
    async with asyncio.timeout(5):
        registro = ModuloRegistro()
        with patch.object(registro.logger, "error") as mock_logger:
            nucleus.config["bloques"] = [{"id": "invalid_block", "canal": -1, "entidades": 500}]  # Canal inválido
            await registro.inicializar(nucleus)
            assert "invalid_block" not in registro.bloques
            assert nucleus.publicar_alerta.called
            assert mock_logger.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_inicializar(nucleus):
    """Prueba la inicialización de ModuloSincronizacion."""
    async with asyncio.timeout(5):
        sincronizacion = ModuloSincronizacion()
        with patch.object(sincronizacion.logger, "info") as mock_logger:
            await sincronizacion.inicializar(nucleus)
            assert sincronizacion.nucleus == nucleus
            assert mock_logger.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades(nucleus):
    """Prueba la redirección de entidades en ModuloSincronizacion."""
    async with asyncio.timeout(5):
        sincronizacion = ModuloSincronizacion()
        await sincronizacion.inicializar(nucleus)
        registro = nucleus.modules["registro"]
        async def test_func(): return {"valor": 0.7}
        entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(1000)]
        bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], nucleus=nucleus)
        bloque2 = BloqueSimbiotico("block2", 1, entidades[500:], nucleus=nucleus)
        bloque1.fitness = 0.2  # Bajo fitness
        bloque2.fitness = 0.9  # Alto fitness
        registro.bloques["block1"] = bloque1
        registro.bloques["block2"] = bloque2
        with patch.object(sincronizacion.logger, "info") as mock_logger:
            await sincronizacion.redirigir_entidades("block1", "block2", 200, 1)
            assert len(bloque1.entidades) == 300
            assert len(bloque2.entidades) == 700
            assert nucleus.publicar_alerta.called
            assert mock_logger.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades_error(nucleus):
    """Prueba la redirección de entidades con bloques inexistentes."""
    async with asyncio.timeout(5):
        sincronizacion = ModuloSincronizacion()
        await sincronizacion.inicializar(nucleus)
        with patch.object(sincronizacion.logger, "error") as mock_logger:
            await sincronizacion.redirigir_entidades("block1", "block2", 200, 1)
            assert mock_logger.called
            assert not nucleus.publicar_alerta.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_adaptar_bloque_fusionar(nucleus):
    """Prueba la fusión de bloques en ModuloSincronizacion."""
    async with asyncio.timeout(5):
        sincronizacion = ModuloSincronizacion()
        await sincronizacion.inicializar(nucleus)
        registro = nucleus.modules["registro"]
        async def test_func(): return {"valor": 0.7}
        entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(1000)]
        bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], nucleus=nucleus)
        bloque2 = BloqueSimbiotico("block2", 1, entidades[500:], nucleus=nucleus)
        bloque1.fitness = 0.1  # Bajo fitness
        bloque2.fitness = 0.6  # Alto fitness
        registro.bloques["block1"] = bloque1
        registro.bloques["block2"] = bloque2
        with patch.object(sincronizacion.logger, "info") as mock_logger:
            await sincronizacion.adaptar_bloque("block1", carga=0.1)
            assert "block1" not in registro.bloques
            assert "block2" not in registro.bloques
            assert any("fus_" in bid for bid in registro.bloques)
            assert nucleus.publicar_alerta.called
            assert mock_logger.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_ejecucion_inicializar(nucleus):
    """Prueba la inicialización de ModuloEjecucion."""
    async with asyncio.timeout(5):
        ejecucion = ModuloEjecucion()
        with patch.object(ejecucion.logger, "info") as mock_logger:
            await ejecucion.inicializar(nucleus)
            assert ejecucion.nucleus == nucleus
            assert mock_logger.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas(nucleus):
    """Prueba el encolado de tareas en ModuloEjecucion."""
    async with asyncio.timeout(5):
        ejecucion = ModuloEjecucion()
        await ejecucion.inicializar(nucleus)
        with patch.object(ejecucion, "ejecutar_bloque_task") as mock_task, patch.object(ejecucion.logger, "info") as mock_logger:
            await ejecucion.ejecutar()
            assert mock_task.delay.called
            assert nucleus.publicar_alerta.called
            assert mock_logger.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas_error(nucleus):
    """Prueba el encolado de tareas con error."""
    async with asyncio.timeout(5):
        ejecucion = ModuloEjecucion()
        await ejecucion.inicializar(nucleus)
        with patch.object(ejecucion, "ejecutar_bloque_task", side_effect=Exception("Task error")), patch.object(ejecucion.logger, "error") as mock_logger:
            await ejecucion.ejecutar()
            assert nucleus.publicar_alerta.called
            assert mock_logger.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_auditoria_inicializar(nucleus):
    """Prueba la inicialización de ModuloAuditoria."""
    async with asyncio.timeout(5):
        auditoria = ModuloAuditoria()
        with patch.object(auditoria.logger, "info") as mock_logger:
            await auditoria.inicializar(nucleus)
            assert auditoria.nucleus == nucleus
            assert auditoria.detector is not None
            assert mock_logger.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias(nucleus, mock_postgresql):
    """Prueba la detección de anomalías en ModuloAuditoria."""
    async with asyncio.timeout(5):
        auditoria = ModuloAuditoria()
        await auditoria.inicializar(nucleus)
        with patch("psycopg2.connect", return_value=mock_postgresql):
            mock_postgresql.cursor.return_value.fetchall.side_effect = [
                [(100, 0.9), (200, 0.1)],  # Datos
                [("block1"), ("block2")]    # IDs
            ]
            await auditoria.detectar_anomalias()
            assert mock_postgresql.cursor.called
            assert nucleus.publicar_alerta.called
        await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias_error(nucleus, mock_postgresql):
    """Prueba la detección de anomalías con error en PostgreSQL."""
    async with asyncio.timeout(5):
        auditoria = ModuloAuditoria()
        await auditoria.inicializar(nucleus)
        with patch("psycopg2.connect", return_value=mock_postgresql):
            mock_postgresql.cursor.side_effect = Exception("Database error")
            await auditoria.detectar_anomalias()
            assert mock_postgresql.cursor.called
            assert not nucleus.publicar_alerta.called
        await nucleus.detener()
