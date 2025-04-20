import logging
import random
from typing import Dict
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from pydantic import ValidationError
from corec.core import PluginBlockConfig
from corec.entities import crear_entidad


class ModuloRegistro(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloRegistro")
        self.nucleus = None
        self.bloques: Dict[str, BloqueSimbiotico] = {}

    async def inicializar(self, nucleus):
        """Inicializa el módulo de registro."""
        self.nucleus = nucleus
        self.logger.info("[Registro] Iniciando inicialización")
        try:
            bloques_conf = nucleus.config.get("bloques", [])
            self.logger.info(f"[Registro] Procesando {len(bloques_conf)} bloques de configuración: {bloques_conf}")
            for bloque_conf in bloques_conf:
                self.logger.debug(f"[Registro] Procesando bloque: {bloque_conf}")
                try:
                    config = PluginBlockConfig(**bloque_conf)
                    self.logger.debug(f"[Registro] Configuración validada: id={config.id}, canal={config.canal}, entidades={config.entidades}")
                    entidades = [crear_entidad(f"ent_{i}", config.canal, lambda: {"valor": random.uniform(0, 1)}) for i in range(config.entidades)]
                    self.logger.debug(f"[Registro] Creadas {len(entidades)} entidades para bloque {config.id}")
                    self.logger.debug(f"[Registro] Intentando crear BloqueSimbiotico para {config.id}")
                    bloque = BloqueSimbiotico(config.id, config.canal, entidades, self.nucleus, max_size_mb=1.0)
                    self.logger.debug(f"[Registro] BloqueSimbiotico creado para {config.id}: {bloque}")
                    self.bloques[config.id] = bloque
                    self.logger.debug(f"[Registro] Bloque asignado a self.bloques[{config.id}]: {self.bloques[config.id]}")
                    self.logger.info(f"[Registro] Bloque '{config.id}' registrado")
                    await self.nucleus.publicar_alerta({
                        "tipo": "bloque_registrado",
                        "bloque_id": config.id,
                        "entidades": config.entidades,
                        "canal": config.canal,
                        "timestamp": random.random()
                    })
                except ValidationError as e:
                    self.logger.error(f"[Registro] Configuración inválida para bloque {bloque_conf.get('id', 'desconocido')}: {e}")
                    await self.nucleus.publicar_alerta({
                        "tipo": "error_registro",
                        "bloque_id": bloque_conf.get("id", "desconocido"),
                        "mensaje": str(e),
                        "timestamp": random.random()
                    })
                except Exception as e:
                    self.logger.error(f"[Registro] Error inesperadoերը

System: ¡Seguimos avanzando, amigo! La salida de `pytest` muestra que estamos muy cerca: el fallo en `test_modulo_registro_inicializar` ahora es por la aserción `assert mock_redis_url.called` (`False`), lo que indica que `aioredis.from_url` no se llamó durante `ModuloRegistro.inicializar`. Esto es esperado, ya que `corec/modules/registro.py` no interactúa con Redis, y la aserción es innecesaria, similar al problema previo con `mock_init_db.called`. La buena noticia es que `mock_bloque.called` y otras aserciones pasaron, confirmando que el mock de `BloqueSimbiotico` está funcionando correctamente con `patch("corec.modules.registro.BloqueSimbiotico")`.

Con **42 tests** en total (21 en `tests/test_modules.py` [20 pasados + 1 fallido] + 22 de `tests/test_blocks.py` [7], `tests/test_nucleus.py` [11], `tests/test_plugin.py` [4]), falta **1 test** para los 43 esperados, probablemente en `tests/test_entities.py` u otro archivo no ejecutado. Dado que estás trabajando desde un teléfono y quieres **arreglar todo**, nos enfocaremos en:
1. Corregir el fallo en `test_modulo_registro_inicializar` eliminando la aserción `assert mock_redis_url.called` y el mock de `aioredis.from_url` para evitar errores `F841`.
2. Verificar que el test pase y que no surjan nuevos fallos.
3. Identificar el test faltante ejecutando `pytest --collect-only`.
4. Confirmar que los errores de Flake8 estén resueltos.

La solución es eliminar la aserción `assert mock_redis_url.called` y el mock de `aioredis.from_url`, ya que no son relevantes para `ModuloRegistro.inicializar`. También verificaremos `corec/modules/registro.py` para asegurar que no dependa de Redis. Las instrucciones serán claras, con archivos completos y pasos manejables para facilitar la edición desde tu teléfono, incluyendo opciones para usar la interfaz de GitHub. ¡Vamos a resolver este último fallo y alcanzar los 43 tests! 🚀

---

### Análisis del fallo

#### 1. Fallo en `tests/test_modules.py`

- **`test_modulo_registro_inicializar`**:
  - **Fallo**: `assert mock_redis_url.called` (`False`).
  - **Detalles clave**:
    - Configuración: `[{"id": "test_block", "canal": 1, "entidades": 1000}]`.
    - `registro.bloques`: `['test_block']`, indicando que el bloque se registra.
    - `mock_bloque.called`: No aparece en el error, lo que implica que esta aserción pasó.
    - `mock_redis_url.called`: `False`, indicando que `aioredis.from_url` no se llamó.
    - No se lanza ninguna excepción (el bloque `try`/`except` no falla).
    - Mockeamos `aioredis.from_url` (`patch("aioredis.from_url", new=AsyncMock()) as mock_redis_url`), pero `ModuloRegistro.inicializar` no lo usa.
  - **Causa**:
    - `ModuloRegistro.inicializar` no invoca `aioredis.from_url`, ya que su lógica se centra en crear instancias de `BloqueSimbiotico` a partir de `nucleus.config["bloques"]` y no interactúa con Redis.
    - La aserción `assert mock_redis_url.called` es innecesaria, similar al problema anterior con `mock_init_db.called`. `aioredis.from_url` es relevante para otros componentes (por ejemplo, `CoreCNucleus.inicializar` en `corec/nucleus.py`), pero no para este test.
  - **Solución**:
    - Eliminar la aserción `assert mock_redis_url.called` y el mock de `aioredis.from_url` (`patch("aioredis.from_url", new=AsyncMock()) as mock_redis_url`) para evitar errores `F841`.
    - Verificar que todas las demás aserciones (`mock_bloque.called`, `self.bloques["test_block"] == mock_bloque_instance`, etc.) pasen.
    - Confirmar que `corec/modules/registro.py` no depende de Redis.

#### 2. Estado general
- **Tests totales**: 42 tests (21 en `tests/test_modules.py` [20 pasados + 1 fallido] + 22 de otros archivos). Falta **1 test** para los 43, probablemente en `tests/test_entities.py` u otro archivo no ejecutado.
- **Progreso**: Resolver los fallos previos (`mock_bloque.called`, `mock_init_db.called`) es un gran avance. El fallo actual es una aserción innecesaria, y eliminarla debería permitir que `test_modulo_registro_inicializar` pase.
- **Problemas clave**:
  - La aserción `assert mock_redis_url.called` no es relevante para `ModuloRegistro.inicializar`.
  - Falta identificar el test número 43.

#### 3. Verificación de Flake8
- El error `F841` previo (`mock_init_db`) se resolvió. Eliminar el mock de `aioredis.from_url` evitará un nuevo error `F841` para `mock_redis_url`.

---

### Corrección del fallo

Nos enfocaremos en corregir el fallo en `test_modulo_registro_inicializar` eliminando la aserción `assert mock_redis_url.called` y el mock de `aioredis.from_url`. Verificaremos que el test pase y ejecutaremos `pytest --collect-only` para encontrar el test faltante.

#### Paso 1: Corregir `tests/test_modules.py`
Eliminaremos la aserción `assert mock_redis_url.called` y el mock de `aioredis.from_url`.

**`tests/test_modules.py`** (versión corregida):
```python
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad


@pytest.mark.asyncio
async def test_modulo_registro_inicializar(nucleus):
    """Prueba la inicialización de ModuloRegistro."""
    registro = ModuloRegistro()
    with patch("corec.modules.registro.BloqueSimbiotico") as mock_bloque, \
         patch.object(registro.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        # Configurar mock_bloque para devolver un objeto válido
        mock_bloque_instance = MagicMock()
        mock_bloque.return_value = mock_bloque_instance
        nucleus.config["bloques"] = [{"id": "test_block", "canal": 1, "entidades": 1000}]
        try:
            await asyncio.wait_for(registro.inicializar(nucleus), timeout=5)
        except Exception as e:
            pytest.fail(f"Excepción inesperada durante inicialización: {e}")
        assert mock_bloque.called, f"mock_bloque no fue llamado. Config: {nucleus.config['bloques']}, Bloques registrados: {list(registro.bloques.keys())}, Mock calls: {mock_bloque.mock_calls}, Bloque registrado: {registro.bloques.get('test_block')}"
        assert "test_block" in registro.bloques
        assert registro.bloques["test_block"] == mock_bloque_instance
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque(nucleus):
    """Prueba el registro de un bloque en ModuloRegistro."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(registro.inicializar(nucleus), timeout=5)
        await asyncio.wait_for(registro.registrar_bloque("new_block", 2, 500), timeout=5)
        assert "new_block" in registro.bloques
        assert registro.bloques["new_block"].canal == 2
        assert len(registro.bloques["new_block"].entidades) == 500
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_registro_registrar_bloque_config_invalida(nucleus):
    """Prueba el registro de un bloque con configuración inválida."""
    registro = ModuloRegistro()
    with patch.object(registro.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        nucleus.config["bloques"] = [{"id": "invalid_block", "canal": -1, "entidades": 500}]
        await asyncio.wait_for(registro.inicializar(nucleus), timeout=5)
        assert "invalid_block" not in registro.bloques
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_inicializar(nucleus):
    """Prueba la inicialización de ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    with patch.object(sincronizacion.logger, "info") as mock_logger:
        await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
        assert sincronizacion.nucleus == nucleus
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades(nucleus):
    """Prueba la redirección de entidades en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], nucleus=nucleus)
    bloque2 = BloqueSimbiotico("block2", 1, entidades[500:], nucleus=nucleus)
    bloque1.fitness = 0.2
    bloque2.fitness = 0.9
    registro.bloques["block1"] = bloque1
    registro.bloques["block2"] = bloque2
    with patch.object(sincronizacion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(sincronizacion.redirigir_entidades("block1", "block2", 200, 1), timeout=5)
        assert len(bloque1.entidades) == 300
        assert len(bloque2.entidades) == 700
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_redirigir_entidades_error(nucleus):
    """Prueba la redirección de entidades con bloques inexistentes."""
    sincronizacion = ModuloSincronizacion()
    await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
    with patch.object(sincronizacion.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(sincronizacion.redirigir_entidades("block1", "block2", 200, 1), timeout=5)
        assert mock_logger.called
        assert not mock_alerta.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_sincronizacion_adaptar_bloque_fusionar(nucleus):
    """Prueba la fusión de bloques en ModuloSincronizacion."""
    sincronizacion = ModuloSincronizacion()
    await asyncio.wait_for(sincronizacion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(1000)]
    bloque1 = BloqueSimbiotico("block1", 1, entidades[:500], nucleus=nucleus)
    bloque2 = BloqueSimbiotico("block2", 1, entidades[500:], nucleus=nucleus)
    bloque1.fitness = 0.1
    bloque2.fitness = 0.6
    registro.bloques["block1"] = bloque1
    registro.bloques["block2"] = bloque2
    with patch.object(sincronizacion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(sincronizacion.adaptar_bloque("block1", carga=0.1), timeout=5)
        assert "block1" not in registro.bloques
        assert "block2" not in registro.bloques
        assert any("fus_" in bid for bid in registro.bloques)
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_ejecucion_inicializar(nucleus):
    """Prueba la inicialización de ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    with patch.object(ejecucion.logger, "info") as mock_logger:
        await asyncio.wait_for(ejecucion.inicializar(nucleus), timeout=5)
        assert ejecucion.nucleus == nucleus
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas(nucleus):
    """Prueba el encolado de tareas en ModuloEjecucion."""
    ejecucion = ModuloEjecucion()
    await asyncio.wait_for(ejecucion.inicializar(nucleus), timeout=5)
    registro = ModuloRegistro()
    nucleus.modules["registro"] = registro
    async def test_func(): return {"valor": 0.7}
    entidades = [crear_entidad(f"m{i}", 1, test_func) for i in range(100)]
    bloque = BloqueSimbiotico("test_block", 1, entidades, nucleus=nucleus)
    registro.bloques["test_block"] = bloque
    with patch.object(ejecucion, "ejecutar_bloque_task") as mock_task, \
         patch.object(ejecucion.logger, "info") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(ejecucion.ejecutar(), timeout=5)
        assert mock_task.delay.called
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_ejecucion_encolar_tareas_error(nucleus):
    """Prueba el encolado de tareas con error."""
    ejecucion = ModuloEjecucion()
    await asyncio.wait_for(ejecucion.inicializar(nucleus), timeout=5)
    with patch.object(ejecucion, "ejecutar_bloque_task", side_effect=Exception("Task error")), \
         patch.object(ejecucion.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        await asyncio.wait_for(ejecucion.ejecutar(), timeout=5)
        assert mock_alerta.called
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_auditoria_inicializar(nucleus):
    """Prueba la inicialización de ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    with patch.object(auditoria.logger, "info") as mock_logger:
        await asyncio.wait_for(auditoria.inicializar(nucleus), timeout=5)
        assert auditoria.nucleus == nucleus
        assert auditoria.detector is not None
        assert mock_logger.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias(nucleus, mock_postgresql):
    """Prueba la detección de anomalías en ModuloAuditoria."""
    auditoria = ModuloAuditoria()
    await asyncio.wait_for(auditoria.inicializar(nucleus), timeout=5)
    with patch("psycopg2.connect", return_value=mock_postgresql), \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        mock_postgresql.cursor.return_value.fetchall.side_effect = [
            [(100, 0.9), (200, 0.1)],  # Datos
            [("block1"), ("block2")]    # IDs
        ]
        mock_postgresql.cursor.return_value.__enter__.return_value.execute.side_effect = None
        await asyncio.wait_for(auditoria.detectar_anomalias(), timeout=5)
        assert mock_postgresql.cursor.called
        assert mock_alerta.called
    await nucleus.detener()


@pytest.mark.asyncio
async def test_modulo_auditoria_detectar_anomalias_error(nucleus, mock_postgresql):
    """Prueba la detección de anomalías con error en PostgreSQL."""
    auditoria = ModuloAuditoria()
    await asyncio.wait_for(auditoria.inicializar(nucleus), timeout=5)
    with patch("psycopg2.connect", return_value=mock_postgresql), \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        mock_postgresql.cursor.side_effect = Exception("Database error")
        await asyncio.wait_for(auditoria.detectar_anomalias(), timeout=5)
        assert mock_postgresql.cursor.called
        assert not mock_alerta.called
    await nucleus.detener()
