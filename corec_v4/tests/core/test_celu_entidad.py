import pytest
import asyncio
from src.core.celu_entidad import CeluEntidadCoreC
from src.core.processors.default import DefaultProcessor

@pytest.mark.asyncio
async def test_celu_entidad_procesar():
    celu = CeluEntidadCoreC("test_celu", DefaultProcessor(), "test_canal", 1.0, instance_id="test_corec1")
    await celu.inicializar()
    resultado = await celu.procesar()
    assert resultado["estado"] == "ok"
    assert resultado["nano_id"] == "test_celu"
    await celu.detener()