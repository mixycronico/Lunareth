Documentación Técnica y Profesional de CoreC (Proyecto Genesis)
1. Introducción
1.1 Propósito
CoreC es un sistema distribuido y modular diseñado para gestionar bloques simbióticos (BloqueSimbiotico) y entidades (Entidad) que procesan datos y publican alertas en tiempo real. Forma parte del proyecto Genesis, un framework orientado a la creación de sistemas biomiméticos avanzados. CoreC utiliza tecnologías como Redis para la gestión de streams en tiempo real y PostgreSQL para almacenamiento persistente, y está diseñado para ser extensible mediante plugins dinámicos.
Fecha de Estabilidad: 21 de abril de 2025, 09:48 AM (según el pipeline exitoso). Versión Actual: CoreC Ultimate v1.2 (basado en memorias del 11/04/2025, 11:35). Audiencia: Esta documentación está dirigida a programadores y desarrolladores que deseen entender, contribuir o extender CoreC.
1.2 Características Clave
	•	Procesamiento Distribuido: Gestiona múltiples entidades dentro de bloques simbióticos para procesar datos de forma eficiente.
	•	Gestión de Alertas en Tiempo Real: Utiliza Redis Streams para publicar alertas de eventos como errores, reparaciones y procesamiento de bloques.
	•	Almacenamiento Persistente: Almacena mensajes en PostgreSQL para análisis y auditoría.
	•	Extensibilidad mediante Plugins: Soporta plugins dinámicos como codex, comm_system, y crypto_trading.
	•	Módulos Especializados: Incluye módulos para registro, sincronización, ejecución y auditoría.
	•	Tareas Asíncronas: Integra Celery para la ejecución de tareas en segundo plano.
2. Requisitos del Sistema
2.1 Dependencias
	•	Python: 3.10.17 o superior.
	•	Librerías:
	◦	aioredis: Para la gestión de Redis Streams.
	◦	asyncpg: Para la conexión asíncrona a PostgreSQL.
	◦	psycopg2: Para la inicialización síncrona de PostgreSQL.
	◦	pydantic: Para la validación de configuraciones.
	◦	celery: Para tareas asíncronas.
	◦	redis: Para el backend de Celery.
	◦	pyyaml: Para cargar archivos de configuración YAML.
	◦	pytest, pytest-asyncio: Para pruebas unitarias.
	◦	flake8: Para linting del código.
	•	Servicios Externos:
	◦	Redis: Servidor Redis para streams y tareas (por defecto: localhost:6379).
	◦	PostgreSQL: Base de datos para almacenamiento persistente (por defecto: localhost:5432).
2.2 Configuración del Entorno
	1	Instalar Dependencias: Asegúrate de tener un archivo requirements.txt con las dependencias necesarias: aioredis
	2	asyncpg
	3	psycopg2-binary
	4	pydantic
	5	celery
	6	redis
	7	pyyaml
	8	pytest
	9	pytest-asyncio
	10	flake8
	11	 Instala las dependencias con: pip install -r requirements.txt
	12	
	13	Configurar Redis:
	◦	Asegúrate de que Redis esté corriendo en localhost:6379 (o ajusta la configuración en corec_config.json).
	◦	Configura el usuario y contraseña según tu entorno.
	14	Configurar PostgreSQL:
	◦	Asegúrate de que PostgreSQL esté corriendo en localhost:5432 (o ajusta la configuración).
	◦	Crea una base de datos corec_db: CREATE DATABASE corec_db;
	◦	
	◦	Actualiza las credenciales en corec_config.json.
3. Arquitectura del Sistema
CoreC está diseñado como un sistema modular y distribuido, con un núcleo central (CoreCNucleus) que coordina módulos, bloques, entidades, y plugins. A continuación, se describe la arquitectura y sus componentes principales.
3.1 Diagrama de Arquitectura
graph TD
    A[CoreCNucleus] --> B[ModuloRegistro]
    A --> C[ModuloSincronizacion]
    A --> D[ModuloEjecucion]
    A --> E[ModuloAuditoria]
    B --> F[BloqueSimbiotico]
    F --> G[Entidad]
    A --> H[Redis Streams]
    A --> I[PostgreSQL]
    A --> J[Plugins]
    J --> K[codex]
    J --> L[comm_system]
    J --> M[crypto_trading]
    D --> N[Celery Worker]
3.2 Componentes Principales
3.2.1 CoreCNucleus
	•	Archivo: corec/nucleus.py
	•	Propósito: Es el núcleo central de CoreC. Gestiona la inicialización, ejecución y detención del sistema, coordina módulos, registra plugins, y publica alertas.
	•	Funcionalidades Clave:
	◦	Inicialización: Carga la configuración, inicializa la conexión a PostgreSQL y Redis, y registra bloques y plugins.
	◦	Ejecución Continua: Procesa bloques, encola tareas, realiza auditorías y sincronizaciones.
	◦	Gestión de Plugins: Registra y ejecuta plugins dinámicamente.
	◦	Publicación de Alertas: Envía alertas a Redis Streams para eventos como errores o procesamiento de bloques.
	•	Métodos Principales:
	◦	inicializar(): Inicializa el sistema, carga módulos y registra bloques.
	◦	ejecutar(): Ejecuta el procesamiento continuo de bloques, auditorías y sincronizaciones.
	◦	registrar_plugin(plugin_id, plugin): Registra un plugin y lo asocia con un bloque.
	◦	ejecutar_plugin(plugin_id, comando): Ejecuta un comando en un plugin.
	◦	publicar_alerta(alerta): Publica una alerta en Redis Streams.
	◦	detener(): Detiene el sistema, cerrando conexiones.
3.2.2 BloqueSimbiotico
	•	Archivo: corec/blocks.py
	•	Propósito: Gestiona un conjunto de entidades, procesa datos, repara entidades inactivas, y escribe mensajes en PostgreSQL.
	•	Funcionalidades Clave:
	◦	Procesamiento: Procesa datos de todas las entidades y calcula un fitness promedio.
	◦	Reparación: Reactiva entidades inactivas y reinicia los fallos.
	◦	Persistencia: Escribe mensajes procesados en PostgreSQL.
	•	Métodos Principales:
	◦	procesar(carga): Procesa las entidades y publica una alerta de procesamiento.
	◦	reparar(): Reactiva entidades inactivas y publica una alerta de reparación.
	◦	escribir_postgresql(conn): Escribe los mensajes en PostgreSQL.
3.2.3 Entidad
	•	Archivo: corec/entities.py
	•	Propósito: Representa una unidad básica de procesamiento dentro de un bloque.
	•	Funcionalidades Clave:
	◦	Procesamiento: Ejecuta una función de procesamiento personalizada.
	◦	Estado: Mantiene un estado (activa o inactiva) para controlar su actividad.
	•	Métodos Principales:
	◦	procesar(carga): Procesa datos con la carga dada.
3.2.4 Módulos
	•	ModuloRegistro (corec/modules/registro.py): Registra bloques simbióticos y publica alertas de registro.
	•	ModuloSincronizacion (corec/modules/sincronizacion.py): Redirige entidades entre bloques y fusiona bloques.
	•	ModuloEjecucion (corec/modules/ejecucion.py): Encola y ejecuta tareas de procesamiento de bloques, integrándose con Celery.
	•	ModuloAuditoria (corec/modules/auditoria.py): Detecta anomalías en los bloques y publica alertas.
3.2.5 Plugins
	•	Archivos: plugins/codex/config.json, plugins/comm_system/config.json, plugins/crypto_trading/config.json
	•	Propósito: Extienden la funcionalidad de CoreC. Cada plugin se asocia con un bloque simbiótico y puede manejar comandos personalizados.
	•	Plugins Definidos:
	◦	codex: (Configurado, pero no implementado).
	◦	comm_system: (Configurado, no implementado).
	◦	crypto_trading: (Configurado, pero no implementado).
3.2.6 Otros Componentes
	•	Procesadores (corec/processors.py): Define ProcesadorSensor y ProcesadorFiltro para manejar datos de entidades.
	•	Redis (corec/redis.py): Inicializa el cliente Redis asíncrono.
	•	Serialización (corec/serialization.py): Proporciona funciones para serializar/deserializar mensajes binarios.
	•	Celery Worker (corec/worker.py): Configura Celery para tareas asíncronas.
	•	Base de Datos (corec/db.py): Inicializa la conexión a PostgreSQL y crea la tabla bloques.
3.3 Flujo de Datos
flowchart TD
    A[CoreCNucleus] -->|inicializar| B[ModuloRegistro]
    B -->|registrar_bloque| C[BloqueSimbiotico]
    C -->|procesar| D[Entidad]
    D -->|procesar| E[ProcesadorSensor/Filtro]
    E -->|resultado| C
    C -->|mensajes| F[PostgreSQL]
    C -->|alertas| G[Redis Streams]
    A -->|ejecutar| H[ModuloEjecucion]
    H -->|encolar_bloque| C
    A -->|ejecutar| I[ModuloAuditoria]
    I -->|detectar_anomalias| J[Alertas]
    J --> G
    A -->|ejecutar| K[ModuloSincronizacion]
    K -->|redirigir_entidades| C
    A -->|ejecutar_plugin| L[Plugins]
    L -->|manejar_comando| M[CommSystem]
    M -->|alertas| G
Explicación del Flujo:
	1	CoreCNucleus inicializa el sistema y delega el registro de bloques a ModuloRegistro.
	2	ModuloRegistro crea bloques simbióticos (BloqueSimbiotico).
	3	CoreCNucleus.ejecutar() coordina el procesamiento continuo:
	◦	ModuloEjecucion encola tareas para procesar bloques.
	◦	BloqueSimbiotico procesa datos mediante sus entidades, usando procesadores como ProcesadorSensor o ProcesadorFiltro.
	◦	ModuloAuditoria detecta anomalías y publica alertas.
	◦	ModuloSincronizacion redirige entidades entre bloques.
	4	Los resultados se almacenan en PostgreSQL y las alertas se publican en Redis Streams.
	5	Los plugins (como CommSystem) pueden manejar comandos y publicar alertas adicionales.
4. Configuración y Ejecución
4.1 Configuración
	1	Archivo de Configuración: config/corec_config.json (ver sección de configuración en la documentación anterior).
	2	Base de Datos:
	◦	Asegúrate de que PostgreSQL esté configurado y la tabla bloques esté creada (ver corec/db.py).
	3	Redis:
	◦	Asegúrate de que Redis esté corriendo y accesible.
4.2 Ejecución
El script principal para ejecutar CoreC es run.sh:
run.sh:
#!/bin/bash

# run.sh - Script para ejecutar CoreC dentro del proyecto Genesis

# Colores para mensajes en la terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # Sin color

# Función para mostrar mensajes
log() {
    echo -e "${GREEN}[CoreC] $1${NC}"
}

error() {
    echo -e "${RED}[Error] $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[Advertencia] $1${NC}"
}

# 1. Verificar que Python 3.10+ esté instalado
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+\.\d+')
if [[ -z "$PYTHON_VERSION" ]]; then
    error "Python 3 no está instalado. Por favor, instala Python 3.10 o superior."
    exit 1
fi

if [[ "$(echo $PYTHON_VERSION | grep -oP '^\d+\.\d+')" < "3.10" ]]; then
    error "Se requiere Python 3.10 o superior. Versión actual: $PYTHON_VERSION"
    exit 1
fi

log "Python $PYTHON_VERSION detectado."

# 2. Verificar que las dependencias estén instaladas
if [[ ! -f "requirements.txt" ]]; then
    error "El archivo requirements.txt no existe."
    exit 1
fi

log "Instalando dependencias desde requirements.txt..."
pip3 install -r requirements.txt --quiet
if [[ $? -ne 0 ]]; then
    error "Fallo al instalar las dependencias. Revisa requirements.txt."
    exit 1
fi

# 3. Verificar que Redis esté corriendo
log "Verificando conexión a Redis..."
redis-cli -h localhost -p 6379 ping >/dev/null 2>&1
if [[ $? -ne 0 ]]; then
    error "Redis no está corriendo en localhost:6379. Por favor, inicia Redis."
    exit 1
fi
log "Redis está corriendo."

# 4. Verificar que PostgreSQL esté corriendo
log "Verificando conexión a PostgreSQL..."
PG_PASSWORD=$(grep '"password"' config/corec_config.json | grep -oP '"password":\s*"\K[^"]+')
if ! psql -h localhost -p 5432 -U postgres -d corec_db -c "\q" >/dev/null 2>&1; then
    error "PostgreSQL no está corriendo en localhost:5432 o la base de datos corec_db no existe."
    exit 1
fi
log "PostgreSQL está corriendo."

# 5. Crear tabla 'bloques' si no existe
log "Inicializando la tabla 'bloques' en PostgreSQL..."
python3 -c "
import psycopg2
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CoreCDB')
try:
    conn = psycopg2.connect(dbname='corec_db', user='postgres', password='$PG_PASSWORD', host='localhost', port=5432)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS bloques (
            id VARCHAR(50) PRIMARY KEY,
            canal INTEGER,
            num_entidades INTEGER,
            fitness FLOAT,
            timestamp FLOAT
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()
    logger.info('[DB] Tabla \"bloques\" inicializada')
except Exception as e:
    logger.error(f'[DB] Error inicializando PostgreSQL: {e}')
"

# 6. Ejecutar CoreC
log "Iniciando CoreC..."
python3 run_corec.py

# 7. Mensaje final
log "CoreC detenido."
Pasos para Ejecutar:
	1	Asegúrate de que run.sh tenga permisos de ejecución: chmod +x run.sh
	2	
	3	Ejecuta el script: ./run.sh
	4	
5. Tests y Cobertura
Total de Tests: 43 Resultado: Todos pasaron en 0.25 segundos (según la captura de pantalla del 21/04/2025, 09:48 AM).
Distribución de Tests:
	•	tests/test_blocks.py (7 tests): Pruebas para BloqueSimbiotico.
	•	tests/test_entities.py (4 tests): Pruebas para Entidad.
	•	tests/test_modules.py (17 tests): Pruebas para los módulos.
	•	tests/test_nucleus.py (11 tests): Pruebas para CoreCNucleus.
	•	tests/test_plugin.py (4 tests): Pruebas para plugins.
Ejecutar Tests:
pytest tests/ -v --capture=no
Ejecutar Linting:
flake8 corec/ tests/ --max-line-length=300
6. Lecciones Aprendidas
	•	Manejo de Excepciones Asíncronas: Usar subclases específicas (como EntidadConError) en lugar de mocks dinámicos mejora la robustez.
	•	Estructura de Módulos: Asegurarse de que todos los directorios tengan __init__.py evita problemas de importación.
	•	Linters y CI/CD: Configurar flake8 en el pipeline requiere instalarlo explícitamente y corregir errores de estilo.
	•	Tests y Mocks: Alinear las expectativas de los tests con el comportamiento real del código es clave para evitar fallos.
	•	Ciclo de Vida del Sistema: Implementar métodos como ejecutar() es esencial para garantizar que el sistema sea completamente funcional y no solo se inicialice.
7. Estado Actual
	•	Tests: Los 43 tests pasaron en 0.25 segundos.
	•	Pipeline de CI/CD: Pasó sin errores, incluyendo los pasos de tests y linting.
	•	Estabilidad: CoreC es estable y completamente funcional en el proyecto Genesis, con un ciclo de vida completo que incluye inicialización, ejecución y detención.

Confirmación Final
Esta documentación técnica ahora incluye la implementación del método CoreCNucleus.ejecutar(), asegurando que CoreC sea un sistema completamente funcional con un ciclo de vida continuo. También hemos confirmado la ubicación de run_corec.py en el directorio raíz y ajustado run.sh para ejecutarlo correctamente.
