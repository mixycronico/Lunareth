# tests/test_entities.py
import pytest
import asyncio
from corec.entities import Entidad, crear_entidad

@pytest.mark.asyncio
async def test_entidad_procesar():
    """Prueba el procesamiento de una entidad."""
    entidad = Entidad("ent_1", 1, lambda carga: {"valor": 0.5, "clasificacion": "test", "probabilidad": 0.9})
    result = await entidad.procesar(0.5)
    assert result == {"valor": 0.5, "clasificacion": "test", "probabilidad": 0.9}

@pytest.mark.asyncio
async def test_entidad_procesar_error():
    """Prueba el procesamiento de una entidad con error."""
    entidad = Entidad("ent_1", 1, lambda carga: {"valor": "invalid"})
    result = await entidad.procesar(0.5)
    assert result == {"valor": "invalid"}

def test_procesar_entidad():
    """Prueba la creación y configuración de una entidad."""
    entidad = crear_entidad("ent_1", 1, lambda carga: {"valor": 0.5})
    assert entidad.id == "ent_1"
    assert entidad.canal == 1
    assert entidad.estado == "activa"

@pytest.mark.asyncio
async def test_celu_entidad_procesar():
    """Prueba el procesamiento de una entidad celular."""
    entidad = Entidad("celu_ent_1", 2, lambda carga: {"valor": 0.7, "clasificacion": "celu", "probabilidad": 0.8})
    result = await entidad.procesar(0.7)
    assert result == {"valor": 0.7, "clasificacion": "celu", "probabilidad": 0.8}

def test_procesar_celu_entidad():
    """Prueba la creación y configuración de una entidad celular."""
    entidad = crear_entidad("celu_ent_1", 2, lambda carga: {"valor": 0.7})
    assert entidad.id == "celu_ent_1"
    assert entidad.canal == 2
    assert entidad.estado == "activa"

@pytest.mark.asyncio
async def test_entidad_estado_cambio():
    """Prueba el cambio de estado de una entidad."""
    entidad = Entidad("ent_1", 1, lambda carga: {"valor": 0.5})
    entidad.estado = "inactiva"
    assert entidad.estado == "inactiva"
    entidad.estado = "activa"
    assert entidad.estado == "activa"
