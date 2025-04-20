import pytest
from unittest.mock import patch
from corec.entities import Entidad, crear_entidad


@pytest.mark.asyncio
async def test_entidad_procesar():
    """Prueba el procesamiento de una entidad."""
    entidad = Entidad("ent_1", 1, lambda carga: {"valor": 0.5})
    result = await entidad.procesar(0.5)
    assert result == {"valor": 0.5}


def test_procesar_entidad():
    """Prueba la creación y procesamiento sincrónico de una entidad."""
    entidad = crear_entidad("ent_1", 1, lambda carga: {"valor": 0.5})
    assert entidad.id == "ent_1"
    assert entidad.canal == 1


@pytest.mark.asyncio
async def test_celu_entidad_procesar():
    """Prueba el procesamiento de una entidad celular."""
    entidad = Entidad("celu_ent_1", 2, lambda carga: {"valor": 0.7})
    result = await entidad.procesar(0.7)
    assert result == {"valor": 0.7}


def test_procesar_celu_entidad():
    """Prueba la creación y procesamiento sincrónico de una entidad celular."""
    entidad = crear_entidad("celu_ent_1", 2, lambda carga: {"valor": 0.7})
    assert entidad.id == "celu_ent_1"
    assert entidad.canal == 2
