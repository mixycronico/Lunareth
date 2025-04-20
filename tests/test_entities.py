import pytest
from corec.entities import crear_entidad, procesar_entidad, crear_celu_entidad, procesar_celu_entidad


@pytest.mark.asyncio
async def test_entidad_procesar():
    """Prueba el procesamiento de una entidad estándar."""
    async def test_func(): return {"valor": 0.7}
    entidad = crear_entidad("test_entidad", 1, test_func)
    resultado = await entidad.procesar(carga=0.5)
    assert resultado == {"valor": 0.7}
    assert entidad.estado == "activa"


def test_procesar_entidad():
    """Prueba la función sincrónica procesar_entidad."""
    def test_func(): return {"valor": 0.7}
    entidad = crear_entidad("test_entidad", 1, test_func)
    resultado = procesar_entidad(entidad, carga=0.5)
    assert resultado == {"valor": 0.7}
    assert entidad.estado == "activa"


@pytest.mark.asyncio
async def test_celu_entidad_procesar():
    """Prueba el procesamiento de una entidad celular."""
    async def test_func(): return {"valor": 0.7}
    entidad = crear_celu_entidad("test_celu_entidad", 1, test_func)
    resultado = await entidad.procesar(carga=0.5)
    assert resultado == {"valor": 0.7}
    assert entidad.estado == "activa"
    assert entidad.memoria[0.5] == 0.7


def test_procesar_celu_entidad():
    """Prueba la función sincrónica procesar_celu_entidad."""
    def test_func(): return {"valor": 0.7}
    entidad = crear_celu_entidad("test_celu_entidad", 1, test_func)
    resultado = procesar_celu_entidad(entidad, carga=0.5)
    assert resultado == {"valor": 0.7}
    assert entidad.estado == "activa"
    assert entidad.memoria[0.5] == 0.7
