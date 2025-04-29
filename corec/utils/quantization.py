import pytest
import asyncio
from corec.utils.quantization import escalar
from corec.entities_superpuestas import EntidadSuperpuesta
from corec.blocks import BloqueSimbiotico
from corec.nucleus import CoreCNucleus


@pytest.mark.parametrize("valor,paso,esperado", [
    (0.123, 0.05, 0.1),
    (1.5, 0.05, 1.0),
    (-1.5, 0.05, -1.0),
    (0.0, 0.1, 0.0),
    (0.567, 0.1, 0.6),
    (-0.234, 0.05, -0.25),
])
def test_escalar_valores_validos(valor, paso, esperado):
    """Prueba la función escalar con valores válidos."""
    assert escalar(valor, paso) == pytest.approx(esperado, abs=1e-6)


def test_escalar_paso_invalido():
    """Prueba que escalar lance ValueError para paso inválido."""
    with pytest.raises(ValueError, match="El paso de cuantización debe ser mayor que 0"):
        escalar(0.5, paso=0)
    with pytest.raises(ValueError, match="El paso de cuantización debe ser mayor que 0"):
        escalar(0.5, paso=-0.1)


@pytest.mark.asyncio
async def test_entidad_superpuesta_procesar():
    """Prueba el procesamiento de EntidadSuperpuesta."""
    entidad = EntidadSuperpuesta(
        id="test_ent",
        roles={"rol1": 0.5, "rol2": 0.5},
        quantization_step=0.05,
        min_fitness=0.3,
        mutation_rate=0.1
    )
    resultado = await entidad.procesar(carga=0.5)
    assert "valor" in resultado
    assert "roles" in resultado
    assert resultado["valor"] == pytest.approx(0.5, abs=0.01)  # Cuantizado a 0.05
    assert resultado["roles"] == {"rol1": 0.5, "rol2": 0.5}


@pytest.mark.asyncio
async def test_entidad_superpuesta_mutar_roles():
    """Prueba la mutación de roles en EntidadSuperpuesta."""
    entidad = EntidadSuperpuesta(
        id="test_ent",
        roles={"rol1": 0.5, "rol2": 0.5},
        quantization_step=0.05,
        min_fitness=0.5,
        mutation_rate=1.0  # Forzar mutación
    )
    original_roles = entidad.roles.copy()
    await entidad.mutar_roles(fitness=0.4)
    assert sum(abs(v) for v in entidad.roles.values()) == pytest.approx(1.0, abs=0.01)
    assert entidad.roles != original_roles  # Confirmar que los roles cambiaron


@pytest.mark.asyncio
async def test_bloque_simbiotico_procesar():
    """Prueba el procesamiento de BloqueSimbiotico."""
    nucleus = CoreCNucleus(config_path="config/corec_config.json")
    await nucleus.inicializar()
    entidades = [
        EntidadSuperpuesta(
            f"ent_{i}",
            {"rol1": 0.5, "rol2": 0.5},
            quantization_step=0.05,
            min_fitness=0.3,
            mutation_rate=0.1,
            nucleus=nucleus
        )
        for i in range(2)
    ]
    bloque = BloqueSimbiotico(
        id="test_block",
        canal=1,
        entidades=entidades,
        max_size_mb=10.0,
        nucleus=nucleus,
        quantization_step=0.05,
        max_errores=0.1
    )
    resultado = await bloque.procesar(carga=0.5)
    assert resultado["bloque_id"] == "test_block"
    assert len(resultado["mensajes"]) == 2
    assert "fitness" in resultado
    assert resultado["fitness"] == pytest.approx(0.5, abs=0.01)
    await nucleus.detener()


@pytest.mark.asyncio
async def test_bloque_simbiotico_reparar():
    """Prueba la reparación de BloqueSimbiotico."""
    nucleus = CoreCNucleus(config_path="config/corec_config.json")
    await nucleus.inicializar()
    entidades = [
        EntidadSuperpuesta(
            f"ent_{i}",
            {"rol1": 0.5, "rol2": 0.5},
            quantization_step=0.05,
            min_fitness=0.3,
            mutation_rate=0.1,
            nucleus=nucleus
        )
        for i in range(1)
    ]
    # Simular entidad inactiva
    entidades[0].estado = "inactiva"
    bloque = BloqueSimbiotico(
        id="test_block",
        canal=1,
        entidades=entidades,
        max_size_mb=10.0,
        nucleus=nucleus,
        quantization_step=0.05,
        max_errores=0.1
    )
    await bloque.reparar()
    assert entidades[0].estado == "activa"
    await nucleus.detener()
