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
    with patch("corec.blocks.BloqueSimbiotico") as mock_bloque:
        await registro.inicializar(nucleus)
        assert mock_bloque.called
        assert "test_block" in registro.bloques
        assert nucleus.publicar_alerta.called


@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque(nucleus):
    """Prueba el registro de un bloque en ModuloRegistro."""
    registro = ModuloRegistro()
    await registro.inicializar(nucleus)
    await registro.registrar_bloque("new_block", 2, 500)
    assert "new_block" in registro.bloques
    assert registro.bloques["new_block"].canal == 2
    assert len(registro.bloques["new_block"].entidades) == 500
    assert nucleus.publicar_alerta.called


@pytest.mark.asyncio
async def test_modulo_sincronizacion_inicializar(nucleus):
    """Prueba la inicialización de ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await sincronizacion.inicializar(nucleus)
    assert sincronizacion.nucleus == nucleus
    assert sincronizacion.logger.info.called


@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades(nucleus):
    """Prueba la redirección de entidades en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await sincronizacion.inicializar(nucleus)
    registro = nucleus.modules["registro"]
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], nucleus=nucleus)
    bloque2 = BloqueSimbiotico("block2", 1, entidades[500:], nucleus=nucleus)
    registro.bloques["block1"] = bloque1
    registro.bloques["block2"] = bloque2
    await sincronizacion.redirigir_entidades("block1", "block2", 200, 1)
    assert len(bloque1.entidades) == 300
    assert len(bloque2.entidades) == 700
    assert nucleus.publicar_alerta.called


@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades_error(nucleus):
    """Prueba la redirección de entidades con bloques inexistentes."""
    sincronizacion = ModuloSincronizacion()
    await sincronizacion.inicializar(nucleus)
    await sincronizacion.redirigir_entidades("block1", "block2", 200, 1)
    assert sincronizacion.logger.error.called
    assert not nucleus.publicar_alerta.called


@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas(nucleus):
    """Prueba el encolado de tareas en ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    await ejecucion.inicializar(nucleus)
    with patch.object(ejecucion, "ejecutar_bloque_task") as mock_task:
        await ejecucion.ejecutar()
        assert mock_task.delay.called
        assert nucleus.publicar_alerta.called


@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias(nucleus, mock_postgresql):
    """Prueba la detección de anomalías en ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    await auditoria.inicializar(nucleus)
    with mock_postgresql as conn:
        conn.cursor.return_value.fetchall.side_effect = [
            [(100, 0.9), (200, 0.1)],  # Datos
            [("block1"), ("block2")]    # IDs
        ]
        await auditoria.detectar_anomalias()
        assert conn.cursor.called
        assert nucleus.publicar_alerta.called
      
