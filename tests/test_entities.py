import pytest
import asyncio
from corec.entities import crear_entidad, procesar_entidad, crear_celu_entidad, procesar_celu_entidad
from corec.serialization import deserializar_mensaje


@pytest.mark.asyncio
async def test_crear_entidad():
    """Prueba la creación de una entidad micro."""
    async def test_func(): return {"valor": 0.7}
    entidad = crear_entidad("m123", 1, test_func)
    assert entidad[0] == "m123"
    assert entidad[1] == 1
    assert entidad[2] == test_func
    assert entidad[3] is True


@pytest.mark.asyncio
async def test_procesar_entidad_activa():
    """Prueba el procesamiento de una entidad activa con valor válido."""
    async def test_func(): return {"valor": 0.7}
    entidad = crear_entidad("m123", 1, test_func)
    resultado = await procesar_entidad(entidad, umbral=0.5)
    mensaje = await deserializar_mensaje(resultado)
    assert mensaje["id"] == 123
    assert mensaje["canal"] == 1
    assert mensaje["valor"] == pytest.approx(0.7, rel=1e-6)
    assert mensaje["activo"] is True


@pytest.mark.asyncio
async def test_procesar_entidad_inactiva():
    """Prueba el procesamiento de una entidad inactiva."""
    async def test_func(): return {"valor": 0.7}
    entidad = ("m123", 1, test_func, False)
    resultado = await procesar_entidad(entidad, umbral=0.5)
    mensaje = await deserializar_mensaje(resultado)
    assert mensaje["id"] == 123
    assert mensaje["canal"] == 1
    assert mensaje["valor"] == 0.0
    assert mensaje["activo"] is False


@pytest.mark.asyncio
async def test_procesar_entidad_error():
    """Prueba el procesamiento de una entidad con error."""
    async def test_func(): raise ValueError("Test error")
    entidad = crear_entidad("m123", 1, test_func)
    resultado = await procesar_entidad(entidad, umbral=0.5)
    mensaje = await deserializar_mensaje(resultado)
    assert mensaje["id"] == 123
    assert mensaje["canal"] == 1
    assert mensaje["valor"] == 0.0
    assert mensaje["activo"] is False


@pytest.mark.asyncio
async def test_crear_celu_entidad():
    """Prueba la creación de una entidad celular."""
    async def test_proc(data): return {"valor": data["input"] * 2}
    entidad = crear_celu_entidad("c456", 2, test_proc)
    assert entidad[0] == "c456"
    assert entidad[1] == 2
    assert entidad[2] == test_proc
    assert entidad[3] is True


@pytest.mark.asyncio
async def test_procesar_celu_entidad():
    """Prueba el procesamiento de una entidad celular con datos válidos."""
    async def test_proc(data): return {"valor": data["input"] * 2}
    entidad = crear_celu_entidad("c456", 2, test_proc)
    datos = {"input": 0.4}
    resultado = await procesar_celu_entidad(entidad, datos, umbral=0.5)
    mensaje = await deserializar_mensaje(resultado)
    assert mensaje["id"] == 456
    assert mensaje["canal"] == 2
    assert mensaje["valor"] == pytest.approx(0.8, rel=1e-6)
    assert mensaje["activo"] is True


@pytest.mark.asyncio
async def test_procesar_celu_entidad_inactiva():
    """Prueba el procesamiento de una entidad celular inactiva."""
    async def test_proc(data): return {"valor": data["input"] * 2}
    entidad = ("c456", 2, test_proc, False)
    datos = {"input": 0.4}
    resultado = await procesar_celu_entidad(entidad, datos, umbral=0.5)
    mensaje = await deserializar_mensaje(resultado)
    assert mensaje["id"] == 456
    assert mensaje["canal"] == 2
    assert mensaje["valor"] == 0.0
    assert mensaje["activo"] is False


@pytest.mark.asyncio
async def test_procesar_celu_entidad_error():
    """Prueba el procesamiento de una entidad celular con error."""
    async def test_proc(data): raise ValueError("Test error")
    entidad = crear_celu_entidad("c456", 2, test_proc)
    datos = {"input": 0.4}
    resultado = await procesar_celu_entidad(entidad, datos, umbral=0.5)
    mensaje = await deserializar_mensaje(resultado)
    assert mensaje["id"] == 456
    assert mensaje["canal"] == 2  # Corregido de 1 a 2
    assert mensaje["valor"] == 0.0
    assert mensaje["activo"] is False
