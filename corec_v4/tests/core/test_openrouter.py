import pytest
import asyncio
from src.utils.openrouter import OpenRouterClient

@pytest.mark.asyncio
async def test_openrouter_query_success(monkeypatch):
    async def mock_post(*args, **kwargs):
        class MockResponse:
            status = 200
            async def json(self):
                return {"choices": [{"message": {"content": "Test response"}}]}
        return MockResponse()

    monkeypatch.setattr("aiohttp.ClientSession.post", mock_post)

    client = OpenRouterClient()
    client.enabled = True
    await client.initialize()
    result = await client.query("Test prompt")
    assert result["estado"] == "ok"
    assert result["respuesta"] == "Test response"
    await client.close()

@pytest.mark.asyncio
async def test_openrouter_query_fallback():
    client = OpenRouterClient()
    client.enabled = False
    await client.initialize()
    result = await client.query("Test prompt")
    assert result["estado"] == "fallback"
    assert "No se pudo conectar" in result["respuesta"]
    await client.close()

@pytest.mark.asyncio
async def test_openrouter_analyze_fallback():
    client = OpenRouterClient()
    client.enabled = False
    await client.initialize()
    result = await client.analyze({"data": [1, 2, 3]}, "Test context")
    assert result["estado"] == "ok"
    assert "summary" in result["respuesta"]
    await client.close()