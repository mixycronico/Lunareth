import pytest
import asyncio
from corec.entities import Entidad
from corec.entities_superpuestas import EntidadSuperpuesta
from corec.utils.quantization import escalar

@pytest.mark.asyncio
async def test_entidad_procesar():
    """Prueba el procesamiento de una entidad básica."""
    entidad = Entidad("ent_1", 1, lambda carga: {"valor": 0.5, "clasificacion": "test", "probabilidad": 0.9}, quantization_step=0.05)
    result = await entidad.procesar(0.5)
    assert result == {"valor": 0.5, "clasificacion": "test", "probabilidad": 0.9}

@pytest.mark.asyncio
async def test_entidad_procesar_error():
    """Prueba el procesamiento de una entidad con valor inválido."""
    entidad = Entidad("ent_1", 1, lambda carga: {"valor": "invalid"}, quantization_step=0.05)
    result = await entidad.procesar(0.5)
    assert result == {"valor": "invalid"}

@pytest.mark.asyncio
async def test_entidad_superpuesta_procesar(nucleus):
    """Prueba el procesamiento de una entidad superpuesta."""
    entidad = EntidadSuperpuesta(
        id="ent_1",
        roles={"rol1": 0.5, "rol2": 0.5},
        quantization_step=0.05,
        min_fitness=0.3,
        mutation_rate=0.1,
        nucleus=nucleus
    )
    result = await entidad.procesar(carga=0.5)
    assert "valor" in result
    assert "roles" in result
    assert result["valor"] == pytest.approx(0.5, abs=0.01)
    assert result["roles"] == {"rol1": 0.5, "rol2": 0.5}

@pytest.mark.asyncio
async def test_entidad_superpuesta_mutar_roles(nucleus, monkeypatch):
    """Prueba la mutación de roles en una entidad superpuesta."""
    entidad = EntidadSuperpuesta(
        id="ent_1",
        roles={"rol1": 0.5, "rol2": 0.5},
        quantization_step=0.05,
        min_fitness=0.5,
        mutation_rate=1.0,  # Forzar mutación
        nucleus=nucleus
    )
    original_roles = entidad.roles.copy()
    monkeypatch.setattr("random.random", lambda: 0.1)  # Forzar mutación
    await entidad.mutar_roles(fitness=0.4)
    assert sum(abs(v) for v in entidad.roles.values()) == pytest.approx(1.0, abs=0.01)
    assert entidad.roles != original_roles

@pytest.mark.asyncio
async def test_entidad_estado_cambio():
    """Prueba el cambio de estado de una entidad."""
    entidad = Entidad("ent_1", 1, lambda carga: {"valor": 0.5}, quantization_step=0.05)
    entidad.estado = "inactiva"
    assert entidad.estado == "inactiva"
    entidad.estado = "activa"
    assert entidad.estado == "activa"
