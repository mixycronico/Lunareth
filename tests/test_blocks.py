import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad


@pytest.fixture
def mock_nucleus(mock_redis):
    """Crea un núcleo simulado para pruebas de bloques."""
    nucleus = AsyncMock()
    nucleus.publicar_alerta = AsyncMock()
    nucleus.instance_id = "test_corec"
    nucleus.db_config = {"dbname": "test_db"}
    return nucleus


@pytest.mark.asyncio
async def test_bloque_inicializar(mock_nucleus):
    """Prueba la inicialización de un bloque simbiótico."""
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    assert bloque.id == "test_block"
    assert bloque.canal == 1
    assert len(bloque.entidades) == 100
    assert bloque.fitness == 0.0
    assert bloque.umbral == 0.5
    assert bloque.nucleus == mock_nucleus


@pytest.mark.asyncio
async def test_bloque_procesar(mock_nucleus):
    """Prueba el procesamiento de un bloque simbiótico con carga parcial."""
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    resultado = await bloque.procesar(carga=0.5)
    assert resultado["bloque_id"] == "test_block"
    assert len(resultado["mensajes"]) == 50  # carga=0.5 procesa la mitad
    assert resultado["fitness"] >= 0.0
    assert mock_nucleus.publicar_alerta.called
    assert len(bloque.mensajes) == 50


@pytest.mark.asyncio
async def test_bloque_procesar_con_errores(mock_nucleus):
    """Prueba el procesamiento de un bloque con entidades que fallan."""
    async def test_func_ok(): return {"valor": 0.7}
    async def test_func_error(): raise ValueError("Test error")
    entidades = [crear_entidad(f"m{i}", 1, test_func_ok if i % 2 == 0 else test_func_error) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    resultado = await bloque.procesar(carga=0.5)
    assert resultado["bloque_id"] == "test_block"
    assert len(resultado["mensajes"]) <= 25  # Aproximadamente la mitad de 50 (por errores)
    assert resultado["fitness"] <= 0.0
    assert mock_nucleus.publicar_alerta.called


@pytest.mark.asyncio
async def test_bloque_procesar_datos_invalidos(mock_nucleus):
    """Prueba el procesamiento de un bloque con datos inválidos."""
    async def test_func(): return {"valor": "invalid"}  # Tipo inválido
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    resultado = await bloque.procesar(carga=0.5)
    assert resultado["bloque_id"] == "test_block"
    assert len(resultado["mensajes"]) == 0  # Todos los mensajes son inválidos
    assert resultado["fitness"] <= 0.0
    assert mock_nucleus.publicar_alerta.called


@pytest.mark.asyncio
async def test_bloque_reparar(mock_nucleus):
    """Prueba la autoreparación de un bloque simbiótico."""
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    bloque.mensajes = [{"id": i, "canal": 1, "valor": 0.0, "activo": False} for i in range(10)]
    bloque.fallos = 1
    await bloque.reparar(errores=10)
    assert bloque.fallos == 0
    assert len(bloque.mensajes) == 0
    assert mock_nucleus.publicar_alerta.called
    assert len(bloque.entidades) == 100  # Las entidades se reemplazan, no se eliminan


@pytest.mark.asyncio
async def test_bloque_escribir_postgresql(mock_nucleus, mock_postgresql):
    """Prueba la escritura de un bloque en PostgreSQL."""
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    with patch("psycopg2.connect", return_value=mock_postgresql):
        await bloque.escribir_postgresql(mock_postgresql)  # Pasar mock directamente
        assert mock_postgresql.cursor.called
        assert mock_postgresql.commit.called
        assert mock_nucleus.publicar_alerta.called
        assert len(bloque.mensajes) == 0


@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_error(mock_nucleus, mock_postgresql):
    """Prueba la escritura de un bloque con error en PostgreSQL."""
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    with patch("psycopg2.connect", return_value=mock_postgresql):
        mock_postgresql.cursor.side_effect = Exception("Database error")
        await bloque.escribir_postgresql(mock_postgresql)  # Pasar mock directamente
        assert mock_postgresql.cursor.called
        assert not mock_postgresql.commit.called
        assert mock_nucleus.publicar_alerta.called
        assert len(bloque.mensajes) == 0
