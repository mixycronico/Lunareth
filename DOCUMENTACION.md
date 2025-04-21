# Documentación Técnica de CoreC

**Autor:** Moises Alvarenga y Luna  
**Fecha:** 21 de abril de 2025  
**Licencia:** MIT  

---

## Índice

1. [Introducción](#introducción)  
2. [Arquitectura General](#arquitectura-general)  
3. [Configuración](#configuración)  
4. [Componentes Principales](#componentes-principales)  
   - [CoreCNucleus](#corecnucleus)  
   - [Módulos](#módulos)  
   - [BloqueSimbiotico](#bloquesimbiotico)  
   - [Entidades](#entidades)  
   - [Plugins](#plugins)  
   - [Integración Redis](#integración-redis)  
   - [Persistencia PostgreSQL](#persistencia-postgresql)  
   - [Integración Celery](#integración-celery)  
5. [Flujo de Inicialización y Ejecución](#flujo-de-inicialización-y-ejecución)  
6. [Diagramas UML](#diagramas-uml)  
   - [Diagrama de Componentes](#diagrama-de-componentes)  
   - [Diagrama de Clases](#diagrama-de-clases)  
7. [CI / GitHub Actions](#ci--github-actions)  
8. [Ejemplo de Configuración](#ejemplo-de-configuración)  
9. [Uso y Despliegue](#uso-y-despliegue)  
10. [Pruebas](#pruebas)  
11. [Conclusión](#conclusión)  

---

## Introducción

CoreC es un **núcleo universal** para orquestar aplicaciones distribuidas basadas en bloques simbióticos y plugins. Está diseñado para ser:

- **Modular**: Cada módulo y plugin se inicializa y ejecuta de forma independiente.  
- **Resiliente**: Incluye mecanismos de autoreparación de bloques y auditoría de anomalías.  
- **Escalable**: Soporta decenas de miles de entidades por bloque.  

Esta documentación cubre la configuración, la arquitectura, los principales componentes y el flujo de ejecución, así como ejemplos de uso y el pipeline de CI.

---

## Arquitectura General

```mermaid
graph TD
  subgraph CoreC Nucleus
    CN(CoreCNucleus)
    DB[(PostgreSQL Pool)]
    REDIS[(Redis Client)]
    MODS[Módulos]
    BLOQ[Bloques]
    PLUG[Plugins]
  end

  CN --> DB
  CN --> REDIS
  CN --> MODS
  CN --> BLOQ
  CN --> PLUG

  subgraph Módulos
    MR(ModuloRegistro)
    ME(ModuloEjecucion)
    MS(ModuloSincronizacion)
    MA(ModuloAuditoria)
  end

  MODS --> MR
  MODS --> ME
  MODS --> MS
  MODS --> MA

  subgraph Bloques
    BS(BloqueSimbiotico)
  end

  BLOQ --> BS

  subgraph Plugins
    P(Codex, CommSystem, CryptoTrading…)
  end

  PLUG --> P



⸻

Configuración

El fichero JSON principal config/corec_config.json define:
	•	instance_id: Identificador de la instancia.
	•	db_config: Configuración de PostgreSQL.
	•	redis_config: Configuración de Redis.
	•	bloques: Lista de bloques simbióticos.
	•	plugins: Plugins habilitados y su bloque asociado.

{
  "instance_id": "corec1",
  "db_config": { "dbname": "...", "user": "...", ... },
  "redis_config": { "host": "...", "port": ..., ... },
  "bloques": [ { "id": "enjambre_sensor", "canal": 1, ... } ],
  "plugins": {
    "codex": { "enabled": true, "path": "...", "bloque": { "bloque_id": "codex_block", ... } },
    ...
  }
}



⸻

Componentes Principales

CoreCNucleus
	•	Clase principal que carga configuración, inicializa conexiones y orquesta módulos, bloques y plugins.
	•	Métodos clave:
	•	inicializar(): carga config, crea pool de DB, cliente Redis, inicializa módulos y bloques.
	•	ejecutar(): bucle principal de procesamiento (procesa bloques, auditoría, sincronización).
	•	publicar_alerta(alerta): envía eventos a Redis Streams.
	•	detener(): cierra conexiones y detiene módulos.

Módulos

Cada módulo hereda de ComponenteBase y expone:

async def inicializar(self, nucleus, config=None): ...
async def detener(self): ...

	•	ModuloRegistro
	•	Registra metadatos de bloques (bloques dict).
	•	Publica alerta bloque_registrado.
	•	ModuloEjecucion
	•	Encola bloques para procesar en background.
	•	Publica alerta tarea_encolada.
	•	ModuloSincronizacion
	•	Redirige o adapta entidades entre bloques.
	•	Publica entidades_redirigidas / bloque_adaptado.
	•	ModuloAuditoria
	•	Revisa fitness y anomalías en bloques registrados.
	•	Publica anomalia_detectada.

BloqueSimbiotico
	•	Representa un conjunto de entidades con:
	•	procesar(carga): corre todas las entidades, acumula mensajes y calcula fitness.
	•	reparar(): reactiva entidades inactivas y limpia errores.
	•	escribir_postgresql(conn): persiste mensajes en tabla mensajes.

Entidades
	•	Entidad: objeto ligero con .procesar(carga) que delega a una función de usuario.
	•	Factory: crear_entidad(id, canal, procesar_fn).
	•	Atributo estado → "activa" / "inactiva" usado en autoreparación.

Plugins
	•	Se definen en config["plugins"].
	•	En bootstrap:
	1.	Carga su config.json.
	2.	Importa plugins.<name>.main.
	3.	Llama a inicializar(nucleus, config_plugin).
	•	Registro dinámico via nucleus.registrar_plugin() y ejecución con ejecutar_plugin().

Integración Redis
	•	aioredis.from_url(...) en init_redis().
	•	Alerts via Streams: XADD alertas:<tipo> ....

Persistencia PostgreSQL
	•	Conexión AsyncPG: asyncpg.connect(**db_config).
	•	Tabla bloques creada en init_postgresql().
	•	Inserción de mensajes en escribir_postgresql() de cada bloque.

Integración Celery
	•	App Celery configurada con Redis Broker/Backend.
	•	Incluye tareas en corec.modules.ejecucion.

celery_app = Celery('corec', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')



⸻

Flujo de Inicialización y Ejecución

CoreCNucleus->Config Loader: cargar_config()
CoreCNucleus->PostgreSQL: init_postgresql()
CoreCNucleus->Redis: init_redis()
CoreCNucleus->ModuloRegistro: inicializar()
...
CoreCNucleus->BloqueSimbiotico: crear y registrar bloques
CoreCNucleus->Plugins: load_plugins()
CoreCNucleus->Ejecutar: bucle while True
loop ProcessBloques
    ModuloEjecucion->BloqueSimbiotico: procesar()
    BloqueSimbiotico->PostgreSQL: escribir_postgresql()
end
loop Auditoría
    ModuloAuditoria->Registro: detectar_anomalias()
end
loop Sincronización
    ModuloSincronizacion->BloqueA,B: redirigir_entidades()
end



⸻

Diagramas UML

Diagrama de Componentes

graph LR
  CoreCNucleus --> ModuloRegistro
  CoreCNucleus --> ModuloEjecucion
  CoreCNucleus --> ModuloSincronizacion
  CoreCNucleus --> ModuloAuditoria
  CoreCNucleus --> BloqueSimbiotico
  CoreCNucleus --> Plugin(...)
  BloqueSimbiotico --> Entidad
  ModuloEjecucion --> BloqueSimbiotico
  BloqueSimbiotico --> PostgreSQL
  CoreCNucleus --> Redis

Diagrama de Clases

classDiagram
  class CoreCNucleus {
    - logger: Logger
    - config: dict
    - db_pool
    - redis_client
    - modules: dict
    - plugins: dict
    - bloques: list
    + inicializar()
    + ejecutar()
    + detener()
    + publicar_alerta(alerta)
    + registrar_plugin(id, plugin)
    + ejecutar_plugin(id, comando)
  }

  class BloqueSimbiotico {
    - id: str
    - canal: int
    - entidades: List<Entidad>
    - max_size_mb: float
    - nucleus: CoreCNucleus
    - mensajes: List<dict>
    - fitness: float
    - fallos: int
    + procesar(carga)
    + reparar()
    + escribir_postgresql(conn)
  }

  class Entidad {
    - id: str
    - canal: int
    - estado: str
    - _procesar: Callable
    + procesar(carga) 
  }

  class ModuloBase {
    <<interface>>
    + inicializar(nucleus, config)
    + ejecutar()
    + detener()
  }

  CoreCNucleus --> ModuloBase
  BloqueSimbiotico --> Entidad
  ModuloEjecucion ..> BloqueSimbiotico : encolar_bloque()
  ModuloRegistro ..> BloqueSimbiotico : registrar_bloque()



⸻

CI / GitHub Actions

El flujo /.github/workflows/ci.yml realiza:
	1.	Chequeo de código en main y pull_request.
	2.	Setup de Python 3.10.
	3.	Instalación de dependencias (pip install -r requirements.txt).
	4.	Ejecución de Flake8 (longitud de línea <140).
	5.	Ejecución de pytest.

name: CoreC CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with: { python-version: '3.10' }
      - name: Install deps
        run: pip install --upgrade pip && pip install -r requirements.txt
      - name: Lint
        run: flake8 --max-line-length 140 corec/ tests/
      - name: Test
        run: pytest -q --disable-warnings



⸻

Ejemplo de Configuración

{
  "instance_id": "corec1",
  "db_config": {
    "dbname": "corec_db",
    "user": "postgres",
    "password": "secure_password",
    "host": "localhost",
    "port": 5432
  },
  "redis_config": {
    "host": "localhost",
    "port": 6379,
    "username": "corec_user",
    "password": "secure_password"
  },
  "bloques": [
    {
      "id": "enjambre_sensor",
      "canal": 1,
      "entidades": 10000,
      "max_size_mb": 1,
      "entidades_por_bloque": 1000,
      "autoreparacion": { "max_errores": 0.05, "min_fitness": 0.2 }
    }
  ],
  "plugins": {
    "crypto_trading": {
      "enabled": true,
      "path": "plugins/crypto_trading/config.json",
      "bloque": {
        "bloque_id": "trading_block",
        "canal": 3,
        "entidades": 2000,
        "max_size_mb": 5,
        "max_errores": 0.1,
        "min_fitness": 0.3
      }
    }
  }
}



⸻

Uso y Despliegue
	1.	Clonar repositorio:

git clone https://github.com/mi_org/corec.git
cd corec


	2.	Configurar config/corec_config.json según ejemplo.
	3.	Instalar dependencias:

pip install -r requirements.txt


	4.	Inicializar BD (solo la primera vez):

python -c "from corec.database import init_postgresql; init_postgresql(...)" 


	5.	Arrancar núcleo:

python run_corec.py



⸻

Pruebas
	•	pytest: cubre procesamiento de bloques, reparación, módulos y plugins.
	•	flake8: verifica estilo PEP8 (<140 caracteres).
	•	CI: automatiza lint y tests en cada push/PR.

⸻

Conclusión

CoreC ofrece un framework robusto y flexible para orquestar procesos distribuidos basados en bloques simbióticos y plugins. Gracias a su arquitectura modular y sus mecanismos de autoreparación y auditoría, es ideal para sistemas de alta disponibilidad y escalabilidad.

¡Listo para producción! 🚀

