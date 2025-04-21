Documentación Técnica de CoreC (Proyecto Genesis)

Autor: Moises Alvarenga y Luna 

Fecha: 21 de abril de 2025 

Licencia: MIT

Índice
	1	Introducción
	2	Arquitectura General
	3	Configuración
	4	Componentes Principales
	◦	CoreCNucleus
	◦	Módulos
	◦	BloqueSimbiotico
	◦	Entidades
	◦	Plugins
	◦	Integración Redis
	◦	Persistencia PostgreSQL
	◦	Integración Celery
	5	Flujo de Inicialización y                  Ejecución
	6	Diagramas UML
	◦	Diagrama de Componentes
	◦	Diagrama de Clases
	7	CI / GitHub Actions
	8	Ejemplo de Configuración
	9	Uso y Despliegue
	10	Pruebas
	11	Lecciones Aprendidas
	12	Conclusión

Introducción
CoreC es un núcleo universal para orquestar aplicaciones distribuidas basadas en bloques simbióticos y plugins. Forma parte del proyecto Genesis, un framework diseñado para construir sistemas biomiméticos avanzados. CoreC está diseñado para ser:
	•	Modular: Cada módulo y plugin se inicializa y ejecuta de forma independiente.
	•	Resiliente: Incluye mecanismos de autoreparación de bloques y auditoría de anomalías.
	•	Escalable: Soporta decenas de miles de entidades por bloque.
Versión Actual: CoreC Ultimate v1.2 Fecha de Estabilidad: 21 de abril de 2025 Licencia: MIT
Esta documentación cubre la configuración, la arquitectura, los principales componentes y el flujo de ejecución, así como ejemplos de uso y el pipeline de CI. Está dirigida a programadores que deseen contribuir o extender CoreC, y a administradores de sistemas que gestionen su despliegue.

Arquitectura General
A continuación, se presenta un diagrama de alto nivel que muestra la relación entre los componentes principales de CoreC:
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
    ENT[Entidad]
  end

  BLOQ --> BS
  BS --> ENT

  subgraph Plugins
    P(Codex, CommSystem, CryptoTrading...)
  end

  PLUG --> P

  ME --> N[Celery Worker]
  ME --> BS
  BS --> DB
  CN --> REDIS
  BS --> REDIS
Explicación del Diagrama:
	•	CoreCNucleus: Orquesta el sistema, inicializa conexiones a PostgreSQL y Redis, y coordina módulos, bloques y plugins.
	•	Módulos: Incluyen ModuloRegistro (registra bloques), ModuloEjecucion (encola tareas), ModuloSincronizacion (redirección de entidades), y ModuloAuditoria (detección de anomalías).
	•	Bloques: Cada BloqueSimbiotico contiene múltiples Entidades, procesa datos y escribe resultados en PostgreSQL.
	•	Plugins: Extienden la funcionalidad de CoreC, como Codex, CommSystem, y CryptoTrading.
	•	Celery Worker: Gestiona tareas asíncronas encoladas por ModuloEjecucion.
	•	Redis se usa para alertas en tiempo real.

Configuración
El archivo de configuración principal config/corec_config.json define los parámetros necesarios para inicializar y ejecutar CoreC. Este archivo es cargado por CoreCNucleus durante la inicialización y es utilizado por run.sh y run_corec.py. La estructura del archivo incluye los siguientes campos principales:
	•	instance_id: Identificador único de la instancia de CoreC.
	•	db_config: Configuración de la conexión a PostgreSQL (base de datos, usuario, contraseña, host, puerto).
	•	redis_config: Configuración de la conexión a Redis (host, puerto, usuario, contraseña).
	•	bloques: Lista de bloques simbióticos, cada uno con:
	◦	id: Identificador único del bloque.
	◦	canal: Canal de comunicación (entero positivo).
	◦	entidades: Número total de entidades en el bloque.
	◦	max_size_mb: Tamaño máximo del bloque en MB.
	◦	entidades_por_bloque: Número de entidades por bloque (para particionamiento interno).
	◦	autoreparacion: Configuración de autoreparación (opcional):
	▪	max_errores: Porcentaje máximo de errores permitido antes de reparar.
	▪	min_fitness: Fitness mínimo requerido para considerar el bloque funcional.
	◦	plugin: (Opcional) Nombre del plugin asociado al bloque.
	•	plugins: Diccionario de plugins habilitados, cada uno con:
	◦	enabled: Estado del plugin (true/false).
	◦	path: Ruta al archivo de configuración del plugin (config.json).
	◦	bloque: Configuración del bloque asociado al plugin (incluye bloque_id, canal, entidades, etc.).
La configuración es validada utilizando Pydantic con los modelos PluginBlockConfig y PluginCommand, lo que asegura que los datos sean correctos antes de inicializar el sistema.
Un ejemplo completo de corec_config.json se proporciona en la sección Ejemplo de Configuración.

Componentes Principales
CoreCNucleus
	•	Archivo: corec/nucleus.py
	•	Propósito: Clase principal que carga la configuración, inicializa conexiones y orquesta módulos, bloques y plugins.
	•	Atributos Principales:
	◦	logger: Logger para registrar eventos.
	◦	config: Diccionario con la configuración cargada.
	◦	db_pool: Pool de conexiones a PostgreSQL (AsyncPG).
	◦	redis_client: Cliente Redis asíncrono (aioredis).
	◦	modules: Diccionario de módulos (ModuloRegistro, ModuloEjecucion, etc.).
	◦	plugins: Diccionario de plugins registrados.
	◦	bloques: Lista de bloques simbióticos registrados.
	•	Métodos Clave:
	◦	inicializar(): Carga la configuración, crea el pool de DB, el cliente Redis, inicializa módulos y bloques.
	◦	ejecutar(): Bucle principal de procesamiento (procesa bloques, auditoría, sincronización).
	◦	publicar_alerta(alerta): Envía eventos a Redis Streams.
	◦	detener(): Cierra conexiones y detiene módulos.
	◦	registrar_plugin(plugin_id, plugin): Registra un plugin dinámicamente.
	◦	ejecutar_plugin(plugin_id, comando): Ejecuta un comando en un plugin.
Módulos
Cada módulo hereda de ComponenteBase (corec/core.py) y expone los siguientes métodos:
async def inicializar(self, nucleus, config=None): ...
async def detener(self): ...
	•	ModuloRegistro (corec/modules/registro.py):
	◦	Registra metadatos de bloques en el atributo bloques (diccionario).
	◦	Publica la alerta bloque_registrado en Redis Streams.
	◦	Método clave: registrar_bloque(bloque_id, canal, num_entidades, max_size_mb).
	•	ModuloEjecucion (corec/modules/ejecucion.py):
	◦	Encola bloques para procesar en segundo plano usando Celery.
	◦	Publica la alerta tarea_encolada.
	◦	Método clave: encolar_bloque(bloque).
	•	ModuloSincronizacion (corec/modules/sincronizacion.py):
	◦	Redirige o adapta entidades entre bloques.
	◦	Publica las alertas entidades_redirigidas o bloque_adaptado.
	◦	Métodos clave: redirigir_entidades(bloque_origen, bloque_destino, proporcion, canal), adaptar_bloque(bloque_origen, bloque_destino).
	•	ModuloAuditoria (corec/modules/auditoria.py):
	◦	Revisa el fitness y detecta anomalías en los bloques registrados.
	◦	Publica la alerta anomalia_detectada.
	◦	Método clave: detectar_anomalias().
BloqueSimbiotico
	•	Archivo: corec/blocks.py
	•	Propósito: Representa un conjunto de entidades con las siguientes funcionalidades:
	◦	procesar(carga): Corre todas las entidades, acumula mensajes y calcula el fitness.
	◦	reparar(): Reactiva entidades inactivas y limpia errores.
	◦	escribir_postgresql(conn): Persiste mensajes en la tabla mensajes.
Entidades
	•	Archivo: corec/entities.py
	•	Propósito: Entidad es un objeto ligero con un método procesar(carga) que delega a una función de usuario.
	•	Atributos:
	◦	id: Identificador único.
	◦	canal: Canal de comunicación.
	◦	_estado: Estado interno (activa o inactiva), usado en autoreparación.
	•	Factory: crear_entidad(id, canal, procesar_fn) crea nuevas entidades.
Plugins
	•	Archivos: plugins//config.json
	•	Propósito: Los plugins extienden la funcionalidad de CoreC. Se definen en config["plugins"] y heredan de ComponenteBase.
	•	Ciclo de Vida:
	1	En el arranque (run_corec.py):
	▪	Carga su config.json.
	▪	Importa plugins..main.
	▪	Llama a inicializar(nucleus, config_plugin).
	2	Registro dinámico mediante nucleus.registrar_plugin(id, plugin).
	3	Ejecución de comandos mediante nucleus.ejecutar_plugin(id, comando).
Integración Redis
	•	Archivo: corec/redis.py
	•	Propósito: Gestiona la conexión a Redis para alertas en tiempo real.
	•	Inicialización: Usa aioredis.from_url(...) para crear un cliente asíncrono.
	•	Uso: Publica alertas en Streams con XADD alertas: ....
Persistencia PostgreSQL
	•	Archivo: corec/db.py
	•	Propósito: Gestiona la conexión a PostgreSQL para almacenamiento persistente.
	•	Inicialización: Usa asyncpg.connect(**db_config) para conexiones asíncronas.
	•	Tablas:
	◦	bloques: Almacena metadatos de bloques (creada en init_postgresql()).
	◦	mensajes: Almacena mensajes procesados por bloques (creada dinámicamente por BloqueSimbiotico.escribir_postgresql()).
Integración Celery
	•	Archivo: corec/worker.py
	•	Propósito: Gestiona tareas asíncronas para ModuloEjecucion.
	•	Configuración: celery_app = Celery('corec', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0', include=['corec.modules.ejecucion'])
	•	celery_app.conf.update(
	•	    task_serializer='json',
	•	    accept_content=['json'],
	•	    result_serializer='json',
	•	    timezone='UTC',
	•	    enable_utc=True,
	•	)
	•	
	•	Uso: ModuloEjecucion.encolar_bloque() encola tareas de procesamiento de bloques en segundo plano.
5. Flujo de Inicialización y Ejecución
A continuación, se presenta un diagrama de secuencia que muestra el flujo de inicialización y ejecución de CoreC:
sequenceDiagram
    participant CN as CoreCNucleus
    participant CL as Config Loader
    participant PG as PostgreSQL
    participant R as Redis
    participant MR as ModuloRegistro
    participant ME as ModuloEjecucion
    participant MS as ModuloSincronizacion
    participant MA as ModuloAuditoria
    participant BS as BloqueSimbiotico
    participant PL as Plugins

    CN->>CL: cargar_config()
    CN->>PG: init_postgresql()
    CN->>R: init_redis()
    CN->>MR: inicializar()
    CN->>ME: inicializar()
    CN->>MS: inicializar()
    CN->>MA: inicializar()
    CN->>BS: crear y registrar bloques
    CN->>PL: load_plugins()
    CN->>CN: ejecutar()

    loop Procesamiento de Bloques
        CN->>ME: encolar_bloque()
        ME->>BS: procesar()
        BS->>PG: escribir_postgresql()
    end

    loop Auditoría
        CN->>MA: detectar_anomalias()
        MA->>R: publicar_alerta()
    end

    loop Sincronización
        CN->>MS: redirigir_entidades()
        MS->>BS: redirigir entidades
    end

    CN->>CN: asyncio.sleep(60)
Explicación del Flujo:
	1	Inicialización:
	◦	CoreCNucleus carga la configuración desde corec_config.json.
	◦	Inicializa conexiones a PostgreSQL y Redis.
	◦	Inicializa los módulos (ModuloRegistro, ModuloEjecucion, etc.).
	◦	Crea y registra bloques simbióticos.
	◦	Carga plugins habilitados.
	2	Ejecución:
	◦	Entra en un bucle continuo (ejecutar()).
	◦	ModuloEjecucion encola tareas para procesar bloques.
	◦	BloqueSimbiotico procesa datos y escribe resultados en PostgreSQL.
	◦	ModuloAuditoria detecta anomalías y publica alertas.
	◦	ModuloSincronizacion redirige entidades entre bloques.
	◦	Espera 60 segundos antes del próximo ciclo.
6. Diagramas UML
Diagrama de Componentes
componentDiagram
    [CoreCNucleus] --> [ModuloRegistro]
    [CoreCNucleus] --> [ModuloEjecucion]
    [CoreCNucleus] --> [ModuloSincronizacion]
    [CoreCNucleus] --> [ModuloAuditoria]
    [CoreCNucleus] --> [BloqueSimbiotico]
    [CoreCNucleus] --> [Plugins]
    [BloqueSimbiotico] --> [Entidad]
    [ModuloEjecucion] --> [Celery Worker]
    [ModuloEjecucion] --> [BloqueSimbiotico]
    [BloqueSimbiotico] --> [PostgreSQL]
    [CoreCNucleus] --> [Redis]
    [BloqueSimbiotico] --> [Redis]
    [Plugins] --> [Codex]
    [Plugins] --> [CommSystem]
    [Plugins] --> [CryptoTrading]
Explicación:
	•	CoreCNucleus coordina todos los componentes.
	•	ModuloRegistro y ModuloEjecucion interactúan con BloqueSimbiotico para registrar y procesar bloques.
	•	BloqueSimbiotico contiene múltiples Entidades y escribe datos en PostgreSQL.
	•	Redis se usa para alertas en tiempo real.
	•	Celery Worker gestiona tareas asíncronas.
Diagrama de Clases
classDiagram
    class CoreCNucleus {
        - logger: Logger
        - config: dict
        - db_pool: asyncpg.Pool
        - redis_client: aioredis.Redis
        - modules: dict
        - plugins: dict
        - bloques: list[BloqueSimbiotico]
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
        - entidades: List[Entidad]
        - max_size_mb: float
        - nucleus: CoreCNucleus
        - mensajes: List[dict]
        - fitness: float
        - fallos: int
        + procesar(carga)
        + reparar()
        + escribir_postgresql(conn)
    }

    class Entidad {
        - id: str
        - canal: int
        - _estado: str
        - _procesar: Callable
        + procesar(carga)
    }

    class ComponenteBase {
        <>
        + inicializar(nucleus, config)
        + detener()
    }

    CoreCNucleus --> ComponenteBase
    CoreCNucleus o--> "many" BloqueSimbiotico : bloques
    BloqueSimbiotico --> "many" Entidad : entidades
    ModuloEjecucion ..> BloqueSimbiotico : encolar_bloque()
    ModuloRegistro ..> BloqueSimbiotico : registrar_bloque()
Explicación:
	•	CoreCNucleus contiene una lista de BloqueSimbiotico (bloques).
	•	BloqueSimbiotico contiene múltiples Entidades.
	•	ComponenteBase es una interfaz implementada por módulos y plugins.
	•	ModuloEjecucion y ModuloRegistro interactúan con BloqueSimbiotico mediante métodos específicos.
7. CI / GitHub Actions
El flujo de integración continua está definido en .github/workflows/ci.yml y se ejecuta en cada push y pull_request a la rama main. Realiza los siguientes pasos:
	1	Chequeo de Código: Clona el repositorio usando actions/checkout@v4.
	2	Setup de Python: Configura Python 3.10 con actions/setup-python@v5.
	3	Instalación de Dependencias: Instala las dependencias del proyecto (pip install -r requirements.txt) y flake8.
	4	Linting con Flake8: Verifica el estilo del código con flake8 (longitud máxima de línea: 300 caracteres).
	5	Ejecución de Tests: Ejecuta las pruebas unitarias con pytest, mostrando salida detallada (-v) y sin capturar la salida estándar (--capture=no).
Contenido del Archivo ci.yml:
name: CoreC CI
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install flake8
      - name: Lint
        run: flake8 corec/ tests/ --max-line-length=300
      - name: Test
        run: pytest tests/ -v --capture=no
Notas:
	•	runs-on: ubuntu-latest asegura compatibilidad con la mayoría de entornos.
	•	--capture=no en pytest permite ver la salida en tiempo real, útil para depuración.
	•	--max-line-length=300 en flake8 permite líneas más largas para mantener la legibilidad del código.
8. Ejemplo de Configuración
A continuación, se muestra un ejemplo completo de config/corec_config.json con explicaciones de cada campo:
{
  "instance_id": "corec1",
  "db_config": {
    "dbname": "corec_db",
    "user": "postgres",
    "password": "secure_password_123",
    "host": "localhost",
    "port": 5432
  },
  "redis_config": {
    "host": "localhost",
    "port": 6379,
    "username": "corec_user",
    "password": "redis_password_456"
  },
  "bloques": [
    {
      "id": "enjambre_sensor",
      "canal": 1,
      "entidades": 10000,
      "max_size_mb": 1,
      "entidades_por_bloque": 1000,
      "autoreparacion": {
        "max_errores": 0.05,
        "min_fitness": 0.2
      }
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
Explicación de los Campos:
	•	instance_id: Identificador único de la instancia (ejemplo: "corec1").
	•	db_config:
	◦	dbname: Nombre de la base de datos PostgreSQL (ejemplo: "corec_db").
	◦	user: Usuario de PostgreSQL (ejemplo: "postgres").
	◦	password: Contraseña de PostgreSQL (reemplaza con una contraseña segura).
	◦	host: Host de PostgreSQL (ejemplo: "localhost").
	◦	port: Puerto de PostgreSQL (ejemplo: 5432).
	•	redis_config:
	◦	host: Host de Redis (ejemplo: "localhost").
	◦	port: Puerto de Redis (ejemplo: 6379).
	◦	username: Usuario de Redis (ejemplo: "corec_user").
	◦	password: Contraseña de Redis (reemplaza con una contraseña segura).
	•	bloques:
	◦	id: Identificador único del bloque (ejemplo: "enjambre_sensor").
	◦	canal: Canal de comunicación (ejemplo: 1).
	◦	entidades: Número total de entidades (ejemplo: 10000).
	◦	max_size_mb: Tamaño máximo en MB (ejemplo: 1).
	◦	entidades_por_bloque: Número de entidades por bloque (ejemplo: 1000).
	◦	autoreparacion:
	▪	max_errores: Porcentaje máximo de errores permitido (ejemplo: 0.05).
	▪	min_fitness: Fitness mínimo requerido (ejemplo: 0.2).
	•	plugins:
	◦	crypto_trading:
	▪	enabled: Estado del plugin (true para habilitar).
	▪	path: Ruta al archivo de configuración del plugin.
	▪	bloque: Configuración del bloque asociado (similar a los bloques en bloques).
Nota: Asegúrate de reemplazar "secure_password_123" y "redis_password_456" con contraseñas seguras generadas específicamente para tu entorno.
9. Uso y Despliegue
	1	Clonar el Repositorio: git clone https://github.com/mixycronico/Lunareth.git
	2	cd genesis
	3	
	4	Configurar config/corec_config.json: Ajusta el archivo según el ejemplo proporcionado en la sección Ejemplo de Configuración. Asegúrate de que Redis y PostgreSQL estén corriendo y configurados correctamente.
	5	Instalar Dependencias: pip install -r requirements.txt
	6	
	7	Arrancar el Núcleo: Usa el script run.sh, que verifica dependencias, inicializa la base de datos y arranca CoreC: chmod +x run.sh
	8	./run.sh
	9	
	10	Iniciar el Worker de Celery (Opcional, para tareas asíncronas): En una terminal separada, inicia el worker de Celery para procesar tareas encoladas por ModuloEjecucion: celery -A corec.worker worker --loglevel=info
	11	
Nota: Asegúrate de que Redis y PostgreSQL estén corriendo antes de ejecutar run.sh. El script run.sh se encargará de verificar estas dependencias e inicializar la tabla bloques en PostgreSQL.
10. Pruebas
CoreC incluye un conjunto completo de pruebas unitarias y linting para garantizar la calidad del código:
	•	pytest: Cubre el procesamiento de bloques, reparación, módulos y plugins. Hay un total de 43 pruebas distribuidas en:
	◦	tests/test_blocks.py: 7 pruebas para BloqueSimbiotico.
	◦	tests/test_entities.py: 4 pruebas para Entidad.
	◦	tests/test_modules.py: 17 pruebas para los módulos.
	◦	tests/test_nucleus.py: 11 pruebas para CoreCNucleus.
	◦	tests/test_plugin.py: 4 pruebas para plugins. Comando para Ejecutar:
	•	pytest tests/ -v --capture=no
	•	
	◦	-v: Muestra una salida detallada.
	◦	--capture=no: Permite ver la salida en tiempo real, útil para depuración.
	•	flake8: Verifica el estilo PEP8 con una longitud máxima de línea de 300 caracteres. Comando para Ejecutar: flake8 corec/ tests/ --max-line-length=300
	•	
	•	CI: Automatiza el linting y las pruebas en cada push o pull_request a la rama main, como se detalla en la sección CI / GitHub Actions.
11. Lecciones Aprendidas
	•	Manejo de Excepciones Asíncronas: Usar subclases específicas (como EntidadConError) en lugar de mocks dinámicos mejora la robustez de las pruebas.
	•	Estructura de Módulos: Asegurarse de que todos los directorios tengan __init__.py evita problemas de importación.
	•	Linters y CI/CD: Configurar flake8 en el pipeline requiere instalarlo explícitamente y corregir errores de estilo.
	•	Tests y Mocks: Alinear las expectativas de los tests con el comportamiento real del código es clave para evitar fallos.
	•	Ciclo de Vida del Sistema: Implementar métodos como ejecutar() es esencial para garantizar que el sistema sea completamente funcional y no solo se inicialice.
12. Conclusión
CoreC ofrece un framework robusto y flexible para orquestar procesos distribuidos basados en bloques simbióticos y plugins. Gracias a su arquitectura modular, sus mecanismos de autoreparación y auditoría, y su capacidad para manejar decenas de miles de entidades, es ideal para sistemas de alta disponibilidad y escalabilidad. Ha sido probado exhaustivamente con 43 pruebas unitarias que pasan sin errores y un pipeline de CI/CD completamente funcional, lo que lo hace listo para producción! 🚀
