import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from corec.nucleus import CoreCNucleus
from app import app, get_nucleus

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
async def mock_nucleus():
    nucleus = AsyncMock()
    nucleus.modules = {
        "cognitivo": AsyncMock(
            intuir=AsyncMock(return_value=0.5),
            percibir=AsyncMock(),
            decisiones=[{"opcion": "test", "confianza": 0.5, "timestamp": 12345}],
            yo={"estado": {"confianza": 1.0}},
            generar_metadialogo=AsyncMock(return_value=["Test afirmación"]),
            atencion={"focos": [], "nivel": 0.5}
        )
    }
    return nucleus

@pytest.mark.asyncio
async def test_get_intuicion_valid(client, mock_nucleus, monkeypatch):
    """Prueba la ruta /cognitivo/intuicion con clave válida."""
    monkeypatch.setenv("API_KEY", "test_key")
    with patch("app.get_nucleus", AsyncMock(return_value=mock_nucleus)):
        response = client.get("/cognitivo/intuicion/test", headers={"api_key": "test_key"})
        assert response.status_code == 200
        assert response.json() == {"tipo": "test", "intuicion": 0.5}

@pytest.mark.asyncio
async def test_get_intuicion_invalid_key(client, mock_nucleus):
    """Prueba la ruta /cognitivo/intuicion con clave inválida."""
    with patch("app.get_nucleus", AsyncMock(return_value=mock_nucleus)):
        response = client.get("/cognitivo/intuicion/test", headers={"api_key": "wrong_key"})
        assert response.status_code == 401
        assert response.json() == {"detail": "Invalid API key"}

@pytest.mark.asyncio
async def test_post_percibir_valid(client, mock_nucleus, monkeypatch):
    """Prueba la ruta /cognitivo/percibir con clave válida."""
    monkeypatch.setenv("API_KEY", "test_key")
    datos = {"tipo": "test", "valor": 0.5}
    with patch("app.get_nucleus", AsyncMock(return_value=mock_nucleus)):
        response = client.post("/cognitivo/percibir", json=datos, headers={"api_key": "test_key"})
        assert response.status_code == 200
        assert response.json() == {"status": "Percepción registrada", "tipo": "test"}

@pytest.mark.asyncio
async def test_get_decisiones_valid(client, mock_nucleus, monkeypatch):
    """Prueba la ruta /cognitivo/decisiones con clave válida."""
    monkeypatch.setenv("API_KEY", "test_key")
    with patch("app.get_nucleus", AsyncMock(return_value=mock_nucleus)):
        response = client.get("/cognitivo/decisiones", headers={"api_key": "test_key"})
        assert response.status_code == 200
        assert response.json() == {"decisiones": [{"opcion": "test", "confianza": 0.5, "timestamp": 12345}]}

@pytest.mark.asyncio
async def test_get_yo_valid(client, mock_nucleus, monkeypatch):
    """Prueba la ruta /cognitivo/yo con clave válida."""
    monkeypatch.setenv("API_KEY", "test_key")
    with patch("app.get_nucleus", AsyncMock(return_value=mock_nucleus)):
        response = client.get("/cognitivo/yo", headers={"api_key": "test_key"})
        assert response.status_code == 200
        assert response.json() == {"yo": {"estado": {"confianza": 1.0}}}

@pytest.mark.asyncio
async def test_get_metadialogo_valid(client, mock_nucleus, monkeypatch):
    """Prueba la ruta /cognitivo/metadialogo con clave válida."""
    monkeypatch.setenv("API_KEY", "test_key")
    with patch("app.get_nucleus", AsyncMock(return_value=mock_nucleus)):
        response = client.get("/cognitivo/metadialogo", headers={"api_key": "test_key"})
        assert response.status_code == 200
        assert response.json() == {"afirmaciones": ["Test afirmación"]}

@pytest.mark.asyncio
async def test_get_atencion_valid(client, mock_nucleus, monkeypatch):
    """Prueba la ruta /cognitivo/atencion con clave válida."""
    monkeypatch.setenv("API_KEY", "test_key")
    with patch("app.get_nucleus", AsyncMock(return_value=mock_nucleus)):
        response = client.get("/cognitivo/atencion", headers={"api_key": "test_key"})
        assert response.status_code == 200
        assert response.json() == {"atencion": {"focos": [], "nivel": 0.5}}
