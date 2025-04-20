#Documentación Técnica de CoreC
Índice
	1	Introducción
	2	Arquitectura 2.1. Entidades 2.2. Bloques Simbióticos 2.3. Núcleo (CoreCNucleus) 2.4. Módulos 2.5. Comunicación 2.6. Almacenamiento
	3	Requisitos
	4	Instalación 4.1. Entorno Linux 4.2. Dependencias Python 4.3. Despliegue con Docker
	5	Uso 5.1. Iniciar CoreC 5.2. Monitoreo y Logs 5.3. Ejecutar Pruebas
	6	Configuración 6.1. corec_config.json
	7	Escalabilidad y Rendimiento 7.1. Multi-Nodo 7.2. Optimización de Recursos
	8	Solución de Problemas
	9	Futuras Mejoras
	10	Desarrolladores

1. Introducción
CoreC es un framework bioinspirado diseñado para procesar hasta 1,000,000 de entidades ultraligeras (~1–1.5 KB cada una) en un entorno de baja latencia y alta concurrencia, ocupando aproximadamente 1 GB de RAM. Su arquitectura modular, distribuida y asíncrona ofrece:
	•	Alta concurrencia: Basado en asyncio para ejecución no bloqueante y Celery para tareas distribuidas.
	•	Baja latencia: Procesamiento de bloques simbióticos en <20 ms.
	•	Alta disponibilidad: Uptime >99.99% mediante autoreparación y tolerancia a fallos.
	•	Escalabilidad: Soporte para Redis Cluster y PostgreSQL particionado.
	•	Extensibilidad: Plugins modulares que integran bloques simbióticos para tareas personalizadas.
CoreC beneficia a los plugins al proporcionar una infraestructura robusta para procesamiento paralelo, comunicación eficiente, persistencia optimizada y monitoreo conversacional, permitiendo que cada plugin gestione su propio bloque simbiótico y escale dinámicamente mediante la redirección de entidades no utilizadas.

2. Arquitectura
CoreC se estructura en torno a entidades ultraligeras, bloques simbióticos, un núcleo central (CoreCNucleus), módulos especializados, comunicación basada en streams, y almacenamiento particionado. A continuación, se detalla cada componente.
2.1. Entidades
Las entidades son unidades de procesamiento ultraligeras que ejecutan funciones asíncronas en bloques simbióticos. Existen dos tipos:
	•	MicroCeluEntidadCoreC:
	◦	Estructura: Tupla (id: str, canal: int, función: Callable[[], Awaitable[Dict]], activo: bool).
	◦	Memoria: ~1 KB.
	◦	Función: Ejecuta una coroutine sin parámetros, devolviendo un diccionario con un campo valor: float.
	◦	Uso: Procesamiento autónomo de tareas ligeras (por ejemplo, cálculos de métricas).
	•	CeluEntidadCoreC:
	◦	Estructura: Tupla (id: str, canal: int, procesador: Callable[[Dict], Awaitable[Dict]], activo: bool).
	◦	Memoria: ~1.5 KB.
	◦	Función: Procesa un payload de entrada (Dict), devolviendo un diccionario con valor: float.
	◦	Uso: Procesamiento dependiente de datos externos (por ejemplo, análisis de payloads).
Serialización: Ambas entidades usan struct.pack("!Ibf?", id, canal, valor, activo) para serializar mensajes en un formato binario compacto (uint32, uint8, float32, bool), optimizando el uso de memoria y red.
Beneficio para plugins: Los plugins pueden definir entidades personalizadas para tareas específicas, integrándolas en bloques simbióticos para procesamiento paralelo, con validación de datos mediante pydantic y persistencia eficiente.
2.2. Bloques Simbióticos
Los bloques simbióticos (BloqueSimbiotico) son contenedores de entidades que operan en paralelo, diseñados para procesar datos en canales específicos con autoreparación y persistencia.
	•	Estructura:
	◦	Atributos: id: str, canal: int, entidades: List[MicroCeluEntidadCoreC], fitness: float, mensajes: List[Dict], umbral: float, fallos: int.
	◦	Tamaño:
	▪	Estándar: ~1 MB (~1,000 entidades).
	▪	Inteligente: ~5 MB (~2,000 entidades + modelo ligero, por ejemplo, IsolationForest).
	◦	Memoria: Escalable dinámicamente mediante redirección de entidades.
	•	Flujo de procesamiento (complejidad O(n)):
	1	Selección: n = ceil(len(entidades) * carga), donde carga es un factor entre 0.0 y 1.0.
	2	Ejecución: asyncio.gather ejecuta las entidades seleccionadas en paralelo.
	3	Deserialización: struct.unpack procesa mensajes binarios.
	4	Validación: pydantic valida datos procesados (por ejemplo, id, valor, activo).
	5	Cálculo de fitness: Basado en la desviación estándar de valores y tasa de errores.
	6	Autorreparación: Reemplaza entidades inactivas (activo: False) si fallos >= 2, usando crear_entidad.
	7	Persistencia: Comprime mensajes con Zstandard (nivel 3) e inserta en PostgreSQL (bloques).
	•	Rendimiento:
	◦	Procesamiento: 5–10 ms para un bloque de 1,000 entidades en un servidor de 4 vCPU.
	◦	Latencia de escritura: ~10–20 ms por bloque en PostgreSQL.
	•	Redirección dinámica: Las entidades de bloques inactivos (fitness < 0.3, carga < 0.2) se redirigen a bloques activos (fitness > 0.8) mediante ModuloSincronizacion, maximizando el poder lógico sin aumentar el consumo de recursos.
Beneficio para plugins: Cada plugin puede gestionar un bloque simbiótico dedicado, configurado con entidades específicas para sus tareas, escalando dinámicamente mediante redirección de entidades no utilizadas. Los plugins acceden a la autoreparación, validación de datos, y persistencia optimizada del núcleo.
2.3. Núcleo (CoreCNucleus)
CoreCNucleus es el componente central que orquesta la inicialización, ejecución, y detención del sistema.
	•	Responsabilidades:
	1	Carga de configuración: Lee corec_config.json para configurar db_config, redis_config, bloques, y plugins.
	2	Infraestructura:
	▪	init_postgresql(db_config): Crea tablas particionadas (bloques) e índices.
	▪	init_redis(redis_config): Instancia un cliente aioredis.Redis para streams.
	3	Registro dinámico:
	▪	Módulos: Carga dinámicamente desde corec/modules/*.py.
	▪	Plugins: Carga desde plugins//main.py, asignando un bloque simbiótico por plugin.
	4	Coordinación de bloques: Prioriza tareas de bloques activos mediante coordinar_bloques, optimizando streams de Redis (corec_alerts, corec_commands).
	5	Alertas: Publica alertas en corec_alerts para monitoreo conversacional.
	6	Cierre limpio: Detiene módulos, escribe bloques en PostgreSQL, y cierra Redis.
	•	Ciclo de vida:
	◦	Inicialización: await inicializar() configura infraestructura y registra módulos/plugins.
	◦	Ejecución: await ejecutar() lanza tareas asíncronas para módulos y bloques.
	◦	Detención: await detener() asegura persistencia y cierre de recursos.
Beneficio para plugins: CoreCNucleus proporciona una infraestructura centralizada para registrar bloques simbióticos, coordinar tareas, y publicar alertas, permitiendo que los plugins se enfoquen en lógica específica mientras aprovechan la concurrencia, escalabilidad, y resiliencia del núcleo.
2.4. Módulos
Los módulos son componentes internos que extienden la funcionalidad del núcleo, heredando de ModuloBase e implementando:
	•	async def inicializar(self, nucleus): Configura el módulo.
	•	async def ejecutar(self): Ejecuta tareas recurrentes.
	•	async def detener(self): Cierra recursos.
Módulos actuales:
	•	Registro: Inicializa y registra bloques simbióticos según corec_config.json, integrando bloques de plugins.
	•	Ejecución: Encola procesamiento de bloques con Celery, priorizando bloques de plugins activos.
	•	Sincronización: Fusiona, divide, y redirige entidades entre bloques según fitness y carga, maximizando el poder lógico.
	•	Auditoría: Detecta anomalías en bloques usando IsolationForest, publicando alertas en corec_alerts.
Beneficio para plugins: Los módulos gestionan la infraestructura subyacente (registro, ejecución, sincronización, auditoría), permitiendo que los plugins se integren sin preocuparse por la gestión de recursos, escalabilidad, o tolerancia a fallos.
2.5. Comunicación
CoreC utiliza Redis Streams para una comunicación asíncrona, eficiente, y escalable.
	•	Formato: Mensajes binarios serializados con struct.pack("!Ibf?", id, canal, valor, activo) (~12 bytes por mensaje).
	•	Transporte: Redis Streams (XADD, XREAD) con maxlen configurable y TTL para idempotencia.
	•	Canales:
	◦	Canal 1: Procesamiento genérico (baja prioridad).
	◦	Canal 2: Seguridad (alta prioridad, baja latencia).
	◦	Canal 3: Análisis intensivo (por ejemplo, modelos de IA).
	◦	Canal 4: Alertas críticas y comandos conversacionales.
	◦	Canal 5: Tareas de generación y refactorización.
	•	Streams:
	◦	corec_commands: Comandos entrantes para plugins.
	◦	corec_responses: Respuestas de plugins.
	◦	corec_alerts: Alertas del núcleo y bloques para monitoreo conversacional.
Beneficio para plugins: Los plugins usan streams configurables para comunicación asíncrona, integrándose con CommSystem para interacciones conversacionales y con módulos para coordinar tareas, con baja latencia y alta confiabilidad.
2.6. Almacenamiento
CoreC utiliza PostgreSQL particionado para persistencia optimizada.
	•	Esquema: CREATE TABLE bloques (
	•	    id TEXT PRIMARY KEY,
	•	    canal INT,
	•	    num_entidades INT,
	•	    fitness REAL,
	•	    timestamp DOUBLE PRECISION,
	•	    instance_id TEXT
	•	) PARTITION BY RANGE (timestamp);
	•	
	•	Particionamiento: Particiones mensuales (por ejemplo, bloques_2025_04) para consultas rápidas.
	•	Índices: Compuestos en canal y timestamp DESC para optimizar lecturas.
	•	Compresión: Mensajes JSON comprimidos con Zstandard (nivel 3) antes de insertarse.
Rendimiento:
	•	Escritura: 1–10 operaciones/s por bloque.
	•	Almacenamiento: ~0.5–2 MB/día para 1,000,000 entidades.
Beneficio para plugins: Los plugins persisten datos procesados en bloques simbióticos sin gestionar la infraestructura de almacenamiento, aprovechando particionamiento y compresión para eficiencia.

3. Requisitos
	•	Hardware:
	◦	Mínimo: 2 GB RAM, 2 vCPU, 10 GB disco.
	◦	Recomendado: 4 GB RAM, 4 vCPU, 20 GB disco.
	•	Software:
	◦	Python ≥ 3.9.
	◦	Redis ≥ 7.0.
	◦	PostgreSQL ≥ 13.
	◦	Sistema operativo: Linux (x86_64 / ARM64).
Beneficio para plugins: Los requisitos mínimos permiten ejecutar plugins en entornos personales, mientras que la configuración recomendada soporta cargas intensivas con bloques simbióticos escalados.

4. Instalación
4.1. Entorno Linux
sudo apt update
sudo apt install -y python3-pip python3-dev libpq-dev gcc g++
sudo apt install -y redis-server postgresql postgresql-contrib
4.2. Dependencias Python
pip install --upgrade pip
pip install -r requirements.txt
requirements.txt sugerido:
aioredis>=2.0.0
psycopg2-binary>=2.9.0
pydantic>=1.10.0
zstd>=1.5.0
sklearn>=1.0.0
celery>=5.2.0
prometheus-client>=0.14.0
4.3. Despliegue con Docker
docker-compose.yml:
version: '3.8'
services:
  redis:
    image: redis:7
    ports:
      - "6379:6379"
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: corec_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_password
    ports:
      - "5432:5432"
  corec:
    build: .
    volumes:
      - ./:/app
    working_dir: /app
    command: ["./run.sh"]
    depends_on:
      - redis
      - postgres
Comando:
docker-compose up -d
Beneficio para plugins: La instalación simplificada permite a los desarrolladores de plugins centrarse en la lógica específica, con una infraestructura preconfigurada para Redis y PostgreSQL.

5. Uso
5.1. Iniciar CoreC
./run.sh
Inicia:
	•	CoreCNucleus (orquesta módulos y plugins).
	•	Celery workers (procesamiento distribuido).
	•	Bloques simbióticos (procesamiento paralelo).
5.2. Monitoreo y Logs
	•	Local: tail -f logs/corec.log.
	•	Docker: docker-compose logs -f corec.
Etiquetas clave:
	•	[CoreCNucleus] Inicializado
	•	[ModuloRegistro] Bloque registrado
	•	[Sincronizacion] Entidades redirigidas
5.3. Ejecutar Pruebas
pytest -q
Cobertura:
	•	Serialización/deserialización de entidades.
	•	Procesamiento de bloques simbióticos.
	•	Inicialización y ejecución de módulos.
	•	Coordinación del núcleo.
Beneficio para plugins: Los plugins pueden extender las pruebas para validar su integración con bloques simbióticos, usando el entorno de pruebas del núcleo.

6. Configuración
6.1. corec_config.json
Estructura:
{
  "instance_id": "corec1",
  "db_config": {
    "dbname": "corec_db",
    "user": "postgres",
    "password": "your_password",
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
      "entidades": 980000,
      "max_size_mb": 1,
      "entidades_por_bloque": 1000,
      "autoreparacion": {
        "max_errores": 0.05,
        "min_fitness": 0.2
      }
    },
    {
      "id": "nodo_seguridad",
      "canal": 2,
      "entidades": 1000,
      "max_size_mb": 1,
      "entidades_por_bloque": 1000,
      "autoreparacion": {
        "max_errores": 0.02,
        "min_fitness": 0.5
      }
    },
    {
      "id": "ia_analisis",
      "canal": 3,
      "entidades": 20000,
      "max_size_mb": 5,
      "entidades_por_bloque": 2000
    }
  ],
  "plugins": [
    {
      "name": "example_plugin",
      "enabled": true,
      "config_path": "plugins/example_plugin/config.json",
      "bloque": {
        "bloque_id": "example_block",
        "canal": 4,
        "entidades": 1000,
        "max_size_mb": 3,
        "max_errores": 0.05,
        "min_fitness": 0.5
      }
    }
  ]
}
Campos clave:
	•	bloques: Configura bloques simbióticos globales.
	•	plugins: Define plugins habilitados y sus bloques simbióticos.
	•	bloque: Especifica bloque_id, canal, entidades, max_size_mb, max_errores, y min_fitness para cada plugin.
Beneficio para plugins: Los plugins configuran bloques simbióticos personalizados en corec_config.json, integrándose con la infraestructura del núcleo para procesamiento paralelo y redirección dinámica.

7. Escalabilidad y Rendimiento
7.1. Multi-Nodo
	•	Redis Cluster: Configurable en puertos 7000–7005 para alta disponibilidad y particionamiento de streams.
	•	PostgreSQL: Particionado por rango de timestamp y canal, con réplicas para lecturas distribuidas.
	•	Celery: Workers distribuidos en múltiples nodos, escalando tareas de bloques simbióticos.
7.2. Optimización de Recursos
	•	Memoria:
	◦	Reposo: ~100–150 MB para 10,000 entidades.
	◦	Pico: ~1.2 GB para 1,000,000 entidades con redirección dinámica.
	•	CPU:
	◦	Concurrencia cooperativa: ~0.3–0.5 vCPU para 1,000 transacciones/s.
	◦	Redirección dinámica: ~0.1 vCPU por ciclo (cada 5 minutos).
	•	I/O:
	◦	Redis Streams: ~0.2–0.3 MB/s para 1,000 mensajes/s.
	◦	PostgreSQL: ~1–10 operaciones/s por bloque, con compresión Zstandard.
	•	Redirección de entidades: Maximiza el poder lógico al reasignar entidades de bloques inactivos (fitness < 0.3) a activos (fitness > 0.8), sin aumentar el consumo de recursos (~1–2% CPU adicional).
Beneficio para plugins: Los plugins escalan dinámicamente al aprovechar la redirección de entidades y la infraestructura multi-nodo, procesando tareas intensivas con mínima sobrecarga.

8. Solución de Problemas
	•	ImportError: corec.modules:
	◦	Verificar que corec/modules/__init__.py existe.
	◦	Comprobar permisos en corec/modules.
	•	Redis ConnectionError:
	◦	Ejecutar redis-cli -h host -p port -a password ping.
	◦	Revisar redis_config en corec_config.json.
	•	PostgreSQL OperationalError:
	◦	Comprobar estado: sudo systemctl status postgresql.
	◦	Validar db_config en corec_config.json.
	•	Bloques no registrados:
	◦	Revisar sección bloques y plugins en corec_config.json.
	◦	Comprobar logs: tail -f logs/corec.log para [ModuloRegistro].
	•	Alertas no recibidas:
	◦	Verificar consumo del stream corec_alerts en plugins conversacionales.
	◦	Comprobar conexión Redis: redis-cli monitor.
Beneficio para plugins: Los plugins acceden a logs detallados y alertas conversacionales para diagnosticar problemas, integrándose con la infraestructura de monitoreo del núcleo.

9. Futuras Mejoras
	•	CLI interactivo: Gestión en runtime de bloques, módulos, y plugins (por ejemplo, corec block-redirect ).
	•	Dashboard Grafana/Prometheus: Métricas en tiempo real de fitness, carga, y redirección de entidades.
	•	Evolución orgánica: Mutación dinámica de funciones de entidades basada en fitness y carga.
	•	WebAssembly: Soporte para procesadores personalizados en entidades, optimizando tareas intensivas.
	•	Soporte conversacional avanzado: Integración de flujos guiados para configuración y monitoreo de bloques.
	•	Optimización de redirección: Algoritmos adaptativos para priorizar redirección según patrones de uso.
Beneficio para plugins: Las mejoras futuras permitirán a los plugins implementar lógicas más complejas, aprovechar monitoreo avanzado, y escalar dinámicamente con mínima intervención.

10. Desarrolladores
	•	Moisés Alvarenga
	•	Luna
Contribuciones: Los desarrolladores pueden extender CoreC creando plugins con bloques simbióticos, integrándose con corec_config.json y los streams de Redis (corec_commands, corec_responses, corec_alerts). El código debe cumplir con PEP 8, verificado con Flake8, y documentarse con docstrings claros.

