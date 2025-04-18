import pytest
import asyncio
from src.core.nucleus import CoreCNucleus

@pytest.mark.asyncio
async def test_nucleus_inicializar():
    nucleus = CoreCNucleus(instance_id="test_corec1")
    await nucleus.inicializar()
    assert len(nucleus.modulos) == 4
    await nucleus.detener()

@pytest.mark.asyncio
async def test_nucleus_razonar_fallback(monkeypatch):
    async def mock_analyze(self, data, context):
        return {"estado": "fallback", "respuesta": "No se pudo conectar con OpenRouter."}

    monkeypatch.setattr("src.utils.openrouter.OpenRouterClient.analyze", mock_analyze)

    nucleus = CoreCNucleus(instance_id="test_corec1")
    await nucleus.inicializar()
    result = await nucleus.razonar({"test": 123}, "Test context")
    assert result["estado"] == "ok"
    assert "summary" in result["respuesta"]
    await nucleus.detener()