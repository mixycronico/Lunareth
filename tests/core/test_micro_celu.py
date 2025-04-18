import pytest
import asyncio
from src.core.micro_celu import MicroCeluEntidadCoreC
from src.core.micro_nano_dna import MicroNanoDNA

@pytest.mark.asyncio
async def test_micro_celu_procesar():
    dna = MicroNanoDNA("test", {"min": 0, "max": 1})
    async def funcion():
        return {"valor": 0.5}
    micro = MicroCeluEntidadCoreC("test_micro", funcion, "test_canal", 0.1, dna=dna)
    resultado = await micro.procesar()
    assert resultado["estado"] == "ok"
    assert resultado["resultado"]["valor"] == 0.5
    assert 0 <= resultado["dna"]["fitness"] <= 1