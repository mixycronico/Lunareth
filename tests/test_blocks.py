import pytest
from unittest.mock import AsyncMock, patch
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad


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


@pytest.mark.asyncio
async def test_bloque_procesar(mock_nucleus):
    """Prueba el procesamiento de un bloque simbiótico."""
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    with patch.object(bloque.logger, "debug") as mock_logger, \
         patch.object(mock_nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        resultado = await bloque.procesar(carga=0.5)
        assert resultado["bloque_id"] == "test_block"
        assert len(resultado["mensajes"]) == 100
        assert resultado["fitness"] == 0.7
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_procesar_con_errores(mock_nucleus):
    """Prueba el procesamiento de un bloque con entidades que fallan."""
    async def test_func_ok(): return {"valor": 0.7}
    async def test_func_error(): raise ValueError("Test error")
    entidades = [crear_entidad(f"m{i}", 1, test_func_ok if i % 2 == 0 else test_func_error) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    with patch.object(bloque.logger, "error") as mock_logger, \
         patch.object(mock_nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        resultado = await bloque.procesar(carga=0.5)
        assert resultado["bloque_id"] == "test_block"
        assert len(resultado["mensajes"]) == 100
        assert resultado["fitness"] == 0.35  # 50 entidades OK, 50 fallan
        assert "error" in resultado
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_procesar_datos_invalidos(mock_nucleus):
    """Prueba el procesamiento de un bloque con datos inválidos."""
    async def test_func(): return {"valor": "invalid"}  # Tipo inválido
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    with patch.object(bloque.logger, "debug") as mock_logger, \
         patch.object(mock_nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        resultado = await bloque.procesar(carga=0.5)
        assert resultado["bloque_id"] == "test_block"
        assert len(resultado["mensajes"]) == 100
        assert resultado["fitness"] == 0.0  # No hay valores válidos
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_reparar(mock_nucleus):
    """Prueba la autoreparación de un bloque simbiótico."""
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    entidades[0].estado = "inactiva"
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    with patch.object(bloque.logger, "info") as mock_logger, \
         patch.object(mock_nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await bloque.reparar()
        assert all(entidad.estado == "activa" for entidad in bloque.entidades)
        assert mock_alerta.called
        assert mock_logger.called


@pytest.mark.asyncio
async def test_bloque_escribir_postgresql(mock_nucleus, mock_postgresql):
    """Prueba la escritura de un bloque en PostgreSQL."""
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    bloque.mensajes = [{"entidad_id": f"m{i}", "canal": 1, "valor": 0.7, "timestamp": time.time()} for i in range(10)]
    with patch("psycopg2.connect", return_value=mock_postgresql), \
         patch.object(mock_nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await bloque.escribir_postgresql(mock_postgresql)
        assert mock_postgresql.cursor.called
        assert mock_postgresql.commit.called
        assert mock_alerta.called
        assert len(bloque.mensajes) == 0


@pytest.mark.asyncio
async def test_bloque_escribir_postgresql_error(mock_nucleus, mock_postgresql):
    """Prueba la escritura de un bloque con error en PostgreSQL."""
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, max_size_mb=1, nucleus=mock_nucleus)
    bloque.mensajes = [{"entidad_id": f"m{i}", "canal": 1, "valor": 0.7, "timestamp": time.time()} for i in range(10)]
    with patch("psycopg2.connect", return_value=mock_postgresql), \
         patch.object(mock_nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        mock_postgresql.cursor.side_effect = Exception("Database error")
        await bloque.escribir_postgresql(mock_postgresql)
        assert mock_postgresql.cursor.called
        assert not mock_postgresql.commit.called
        assert mock_alerta.called
