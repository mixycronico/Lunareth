📦 Guía de Desarrollo de Plugins para CoreC

Versión: 1.1
Fecha: Abril 2025
Autores: Moises Alvarenga & Luna

Este documento muestra cómo crear un plugin moderno, robusto y autosuficiente para CoreC, combinando:
    •	Validación de configuración con Pydantic
    •	Métricas Prometheus (Counters, Histograms)
    •	CLI unificado con Click + autocompletado
    •	Sanitización de rutas y esquemas de comandos
    •	Modularidad: cada “processor” en su propio paquete
    •	Conexión a BD asíncrona (asyncpg/aiopg pool)
    •	Tests con pytest‑asyncio y fixtures

⸻

📋 Índice
    1.	Requisitos
    2.	Estructura de Carpetas
    3.	Definir Configuración con Pydantic
    4.	Esquemas de Comando (Pydantic)
    5.	Generator: Scaffold Jinja2
    6.	Manager: Loop, Dispatch y Métricas
    7.	Sanitización y Utilidades
    8.	CLI con Click + Autocompletado
    9.	Conexión Asíncrona a PostgreSQL
    10.	Testing
    11.	Checklist de Publicación

⸻

1. Requisitos
    •	CoreC con plugin‐loader
    •	Python 3.9+
    •	Redis 6+ (Streams)
    •	PostgreSQL 13+
    •	Paquetes:

pip install jinja2 prometheus_client pydantic click click-completion asyncpg pytest pytest-asyncio

Opcional (Reviser): black, pyflakes, transformers, torch

⸻

2. Estructura de Carpetas

plugins/mi_plugin/
├── __init__.py
├── main.py                    # registro / bootstrap
├── config.json                # config/plugin_name
├── requirements.txt           # deps del plugin
├── cli.py                     # CLI con click
├── config.py                  # esquemas Pydantic
├── processors/
│   ├── __init__.py
│   ├── schemas.py             # esquemas de comando
│   ├── generator.py
│   ├── manager.py
│   └── reviser.py             # opcional
├── utils/
│   ├── __init__.py
│   └── helpers.py             # CircuitBreaker, sanitize_path…
└── templates/
    ├── plugin/                # Jinja2 .j2
    ├── react_app/
    └── fastapi_app/



⸻

3. Definir Configuración con Pydantic

from pydantic import BaseModel, Field, SecretStr
from typing import List

class CircuitBreakerConfig(BaseModel):
    max_failures: int = Field(..., gt=0)
    reset_timeout: int = Field(..., gt=0)

class PluginConfig(BaseModel):
    stream_in: str
    stream_out: str
    templates_dir: str
    output_plugins: str
    output_websites: str
    metrics_port: int = Field(8000, gt=0)
    exclude_patterns: List[str] = []
    circuit_breaker: CircuitBreakerConfig

Uso en main.py:

import json
from .config import PluginConfig
raw = json.load(open("plugins/mi_plugin/config.json"))
cfg = PluginConfig(**raw["mi_plugin"])



⸻

4. Esquemas de Comando (Pydantic)

from pydantic import BaseModel, Field
from typing import Literal, Union

class CmdBase(BaseModel):
    action: str

class GeneratePluginParams(BaseModel):
    plugin_name: str = Field(..., min_length=1)

class GenerateWebsiteParams(BaseModel):
    template: Literal["react", "fastapi"]
    project_name: str = Field(..., min_length=1)

class CmdGeneratePlugin(CmdBase):
    action: Literal["generate_plugin"]
    params: GeneratePluginParams

class CmdGenerateWebsite(CmdBase):
    action: Literal["generate_website"]
    params: GenerateWebsiteParams

Cmd = Union[CmdGeneratePlugin, CmdGenerateWebsite]



⸻

5. Generator: Scaffold Jinja2

Incluye copiado de carpetas + render de plantillas:

class Generator:
    def __init__(...): ...
    async def generate_plugin(...): ...
    async def generate_website(...): ...
    def _render_all(...): ...

Uso de run_blocking, sanitize_path, jinja2.render().

⸻

6. Manager: Loop, Dispatch y Métricas

class Manager:
    def __init__(...): ...
    async def init(...): ...
    async def run_loop(...): ...
    async def handle(...): ...
    async def teardown(...): ...

    •	Manejo de Redis Streams: xread, xadd
    •	Prometheus: Counters, Histogram
    •	CircuitBreaker: resiliencia

⸻

7. Sanitización y Utilidades

class CircuitBreaker:
    def check(self) -> bool: ...
    def register_failure(self): ...

def run_blocking(func: Callable, *a, **k): ...
def sanitize_path(base: Path, user_sub: str): ...



⸻

8. CLI con Click + Autocompletado

@click.group()
def cli(): ...

@cli.command()
def gen_plugin(...): ...

@cli.command()
def gen_site(...): ...

Registro CLI en setup.py, autocompletado con click-completion.

⸻

9. Conexión Asíncrona a PostgreSQL

class PluginDB:
    async def connect(...): ...
    async def save_record(...): ...
    async def close(...): ...

Uso de asyncpg.create_pool() + conn.execute().

⸻

10. Testing
    •	pytest‑asyncio
    •	Fixtures temporales para plantillas
    •	Validación de comandos, paths y generador

@pytest.mark.asyncio
async def test_gen_plugin(...): ...



⸻

11. Checklist de Publicación
    •	Tests pasan (pytest)
    •	Lint (flake8) + mypy limpios
    •	CLI funcionando
    •	Documentación actualizada
    •	Plantillas en templates/
    •	config.json de ejemplo

⸻

🌿 Con esta base, tus plugins para CoreC serán ligeros, adaptables y listos para evolución orgánica.