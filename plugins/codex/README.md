# 🌱 Codex Plugin para CoreC

**Versión**: 1.1  
**Fecha**: Abril 2025  
**Autores**: Moises Alvarenga & Luna  
**Licencia**: MIT  

Un plugin de **scaffolding**, **refactorización** y **chatOps** que genera nuevos plugins y aplicaciones web, refactoriza código existente, expone métricas, valida y asegura rutas, y aporta una CLI confortable.

---

## 📋 Índice

1. [Descripción](#descripción)  
2. [Características](#características)  
3. [Requisitos](#requisitos)  
4. [Instalación](#instalación)  
5. [Configuración](#configuración)  
6. [Estructura de Archivos](#estructura-de-archivos)  
7. [API de Comandos Redis](#api-de-comandos-redis)  
8. [Esquemas de Comando (pydantic)](#esquemas-de-comando-pydantic)  
9. [Métricas Prometheus](#métricas-prometheus)  
10. [Refactorización (`revise`)](#refactorización-revise)  
11. [Seguridad y Sanitización](#seguridad-y-sanitización)  
12. [CLI con Autocompletado](#cli-con-autocompletado)  
13. [Extensiones Futuras](#extensiones-futuras)  
14. [Testing](#testing)  

---

## 1. Descripción

**CodexPlugin** lee comandos desde un Redis Stream, los valida con **pydantic**, despacha acciones a un **Generator** y un **Reviser**, expone métricas en HTTP, y asegura rutas para evitar escapes. Además aporta una **CLI** con `click` y autocompletado para invocar estas acciones desde el terminal.

---

## 2. Características

- 🔧 **Scaffolding** de plugins CoreC (`generate_plugin`).  
- 🌐 **Generación** de apps React o FastAPI (`generate_website`).  
- 🛠️ **Refactorización** de código existente (`revise`).  
- 📊 **Métricas** Prometheus: contadores y latencias por acción.  
- ✋ **Validación** de payloads con **pydantic** (acciones y parámetros).  
- 🔒 **Sanitización** de nombres y rutas para evitar inyección de paths.  
- 💻 **CLI** con `click` + `click-completion` para autocompletado.  

---

## 3. Requisitos

- **CoreC** vX.Y con plugin‐loader.  
- **Python 3.9+**  
- **Redis** 6+ (Streams).  
- **Jinja2** (`pip install jinja2`)  
- **Prometheus Client** (`pip install prometheus_client`)  
- **pydantic** (`pip install pydantic`)  
- **click** & **click-completion** (`pip install click click-completion`)  

(Opcional) para refactorización avanzada: **black**, **pyflakes**, **transformers**, **torch**.

---

## 4. Instalación

1. Copia el directorio:
   ```bash
   cp -R plugins/codex /ruta/a/tu/corec/plugins/

  2.	Instala dependencias:

pip install -r plugins/codex/requirements.txt


  3.	Reinicia CoreC:

bash run.sh


  4.	Verifica en logs:

[CodexPlugin] inicializado con revise, métricas y seguridad



⸻

5. Configuración

Edita plugins/codex/config.json:

{
  "codex": {
    "stream_in":         "corec_commands",
    "stream_out":        "corec_responses",
    "templates_dir":     "plugins/codex/templates",
    "output_plugins":    "plugins/",
    "output_websites":   "generated_websites/",
    "exclude_patterns":  ["__pycache__/*","*.pyc"],
    "metrics_port":      8001,
    "stream_timeout":    5000,
    "circuit_breaker": {
      "max_failures": 3,
      "reset_timeout": 900
    }
  }
}

  •	metrics_port: puerto HTTP para /metrics.
  •	stream_timeout: bloqueo en ms para xread.

⸻

6. Estructura de Archivos

plugins/codex/
├── __init__.py
├── main.py                   # Registro e inicialización
├── config.json               # Parámetros (streams, rutas, métricas)
├── requirements.txt          # jinja2, pydantic, prometheus_client, click…
├── cli.py                    # CLI con click + autocompletado
├── processors/
│   ├── __init__.py
│   ├── schemas.py            # Modelos pydantic de comandos
│   ├── generator.py          # Scaffolding de plugins/web
│   ├── reviser.py            # AST/black/AI refactor
│   └── manager.py            # Loop, validación, dispatch, métricas
└── utils/
    ├── __init__.py
    └── helpers.py            # run_blocking, sanitize_path, CircuitBreaker
└── templates/                # Plantillas Jinja2
    ├── plugin/
    ├── react_app/
    └── fastapi_app/



⸻

7. API de Comandos Redis

Envía un JSON al Stream stream_in, recibe respuesta en stream_out:

// Scaffolding de plugin
{ "action":"generate_plugin",
  "params":{"plugin_name":"mi_plugin"} }

// Generación de web
{ "action":"generate_website",
  "params":{"template":"react","project_name":"dashboard"} }

// Refactorización
{ "action":"revise",
  "params":{"file":"plugins/mi_plugin/main.py"} }

Las respuestas incluyen:

{ "status":"ok", "message": "...", "path": "..." }

o { "status":"error", "message": "...", "details": [...] }.

⸻

8. Esquemas de Comando (pydantic)
  •	CmdGeneratePlugin
  •	action: "generate_plugin"
  •	params.plugin_name: alfanumérico, guiones bajos/medias.
  •	CmdGenerateWebsite
  •	action: "generate_website"
  •	params.template: "react" | "fastapi"
  •	params.project_name: alfanumérico.
  •	CmdRevise
  •	action: "revise"
  •	params.file: ruta de archivo (será saneada).

Errores de validación devuelven status:"error" con details.

⸻

9. Métricas Prometheus

Expuestas en http://<host>:metrics_port/metrics:
  •	codex_commands_total{action="..."}
  •	codex_errors_total{action="..."}
  •	codex_latency_seconds_bucket{action="..."}

Configura Prometheus para scrapear este endpoint y crea dashboards en Grafana.

⸻

10. Refactorización (revise)
  1.	Sanitiza la ruta con sanitize_path(base, file).
  2.	Lee el contenido asíncronamente.
  3.	Aplica CodexReviser.revisar_codigo().
  4.	Escribe cambios sólo si hay diferencias.
  5.	Informe de “Sin cambios” o “Archivo revisado”.

⸻

11. Seguridad y Sanitización
  •	Nombres de plugin/project limitados por regex en pydantic.
  •	Rutas resueltas contra un directorio base; evita ../../.
  •	CircuitBreaker detiene la ejecución tras X fallos y auto‑reset.

⸻

12. CLI con Autocompletado

Invoca desde tu shell:

# Genera plugin
codex generate-plugin mi_plugin

# Genera web
codex generate-website react dashboard

# Refactoriza archivo
codex revise plugins/mi_plugin/main.py

  •	Autocompletado de comandos y plantillas con click-completion.
  •	Usa codex --help para ver opciones.

⸻

13. Extensiones Futuras
  •	Integrar CodexMemory para historial de templates.
  •	Soporte de otros lenguajes en reviser (JS, Go…).
  •	Plugins de testing o linting on‑demand.
  •	Motor de versionado de scaffolds y revisiones.

⸻

14. Testing
  •	tests/test_codex.py cubre Generator y Manager.
  •	Añade tests para revise y validación de esquemas.
  •	Ejecuta:

pytest -q



⸻

