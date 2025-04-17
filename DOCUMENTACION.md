Documentación para Desarrolladores de CoreC
CoreC es un sistema distribuido biomimético diseñado para emular la resiliencia, adaptabilidad y modularidad de los organismos vivos. En su Etapa 1, CoreC ofrece un núcleo autónomo (CoreCNucleus) que orquesta entidades celulares (CeluEntidadCoreC, MicroCeluEntidadCoreC) y módulos esenciales (Registro, Ejecución, Sincronización, Auditoría). Con soporte opcional para OpenRouter para análisis y chat, y una infraestructura preparada para plugins, CoreC es una base robusta para aplicaciones escalables.
Esta documentación está dirigida a desarrolladores que deseen entender, configurar, ejecutar, probar, monitorear y extender CoreC. Cubre todos los aspectos técnicos, desde la arquitectura hasta los detalles de implementación, con ejemplos prácticos y mejores prácticas.

Tabla de Contenidos
  1	Introducción
  2	Arquitectura
  3	Componentes Principales
  4	Requisitos Previos
  5	Configuración
  6	Ejecución
  7	Pruebas
  8	Monitoreo
  9	Extensión del Sistema
  10	Mejores Prácticas
  11	Solución de Problemas
  12	Contribución
  13	Licencia

Introducción
CoreC es un sistema distribuido inspirado en la biología, donde el núcleo (CoreCNucleus) actúa como el cerebro, coordinando entidades y módulos para procesar datos, gestionar recursos y mantener la estabilidad. Su diseño biomimético permite:
  •	Autonomía: Funciona sin dependencias externas, con OpenRouter como apoyo opcional.
  •	Modularidad: Preparado para extenderse mediante plugins.
  •	Resiliencia: Soporta fallos con espejos, regeneración de enjambres y fallbacks.
  •	Escalabilidad: Diseñado para instancias distribuidas y alta carga.
En la Etapa 1, CoreC ofrece un núcleo funcional con soporte para plugins vacío, listo para añadir funcionalidades específicas (como CLI, alertas, trading) en la Etapa 2.

Arquitectura
CoreC emula un organismo vivo, con componentes que interactúan como sistemas biológicos:
  •	CoreCNucleus: El cerebro central, que inicializa, coordina y detiene módulos y entidades.
  •	CeluEntidadCoreC: Entidades similares a neuronas que procesan datos para canales específicos.
  •	MicroCeluEntidadCoreC: Micro-células adaptativas con ADN basado en redes neuronales (MicroNanoDNA) para tareas dinámicas.
  •	Módulos:
  ◦	Registro: Gestiona el registro de entidades y enjambres.
  ◦	Ejecución: Orquesta la ejecución de entidades.
  ◦	Sincronización: Maneja balanceo de carga y limpieza de nodos.
  ◦	Auditoría: Monitorea la salud del sistema y genera alertas.
  •	OpenRouter: IA externa opcional para análisis avanzado y capacidades conversacionales, con fallbacks locales.
  •	PluginManager: Infraestructura para cargar plugins dinámicamente, aunque actualmente vacía.
  •	Almacenamiento:
  ◦	PostgreSQL: Almacena nodos, eventos y auditoría.
  ◦	Redis: Gestiona streams para comunicación entre entidades.
La arquitectura sigue principios biomiméticos:
  •	Núcleo como cerebro: CoreCNucleus toma decisiones y coordina.
  •	Entidades como células: Procesan datos de forma distribuida.
  •	Módulos como órganos: Ejecutan funciones especializadas.
  •	Enjambres como tejidos: Grupos de micro-células que colaboran.

Componentes Principales
CoreCNucleus
  •	Ubicación: src/core/nucleus.py
  •	Propósito: Orquesta el sistema, inicializando módulos, entidades y OpenRouter.
  •	Funcionalidades:
  ◦	Inicializa y detiene el sistema (inicializar, detener).
  ◦	Registra entidades (registrar_celu_entidad, registrar_micro_celu_entidad).
  ◦	Proporciona procesadores para canales (get_procesador).
  ◦	Ofrece análisis y chat vía OpenRouter (razonar, responder_chat).
  ◦	Publica alertas en el canal alertas (publicar_alerta).
  •	Dependencias: PostgreSQL, Redis, OpenRouter (opcional).
CeluEntidadCoreC
  •	Ubicación: src/core/celu_entidad.py
  •	Propósito: Procesa datos para un canal específico, similar a una neurona.
  •	Funcionalidades:
  ◦	Inicializa y actualiza configuraciones (inicializar, _actualizar_config).
  ◦	Obtiene datos de la base de datos (obtener_datos).
  ◦	Procesa datos usando un procesador (procesar).
  ◦	Publica resultados en eventos (comunicar).
  ◦	Envía heartbeats a nodos para canales críticos o espejos (_enviar_heartbeat).
  •	Dependencias: PostgreSQL.
MicroCeluEntidadCoreC
  •	Ubicación: src/core/micro_celu.py
  •	Propósito: Ejecuta tareas dinámicas en enjambres, con ADN adaptativo.
  •	Funcionalidades:
  ◦	Procesa datos con una función personalizada (procesar).
  ◦	Entrena una red neuronal (MicroNanoDNA) para optimizar resultados.
  ◦	Publica resultados en Redis streams (comunicar).
  ◦	Se regenera si falla repetidamente.
  •	Dependencias: Redis, PostgreSQL.
MicroNanoDNA
  •	Ubicación: src/core/micro_nano_dna.py
  •	Propósito: Proporciona ADN adaptativo para micro-células.
  •	Funcionalidades:
  ◦	Valida parámetros (_validar_parametros).
  ◦	Mutación y recombinación genética (mutar, recombinar, heredar).
  ◦	Entrena una red neuronal (NanoNeuralNet) (entrenar_red).
  •	Dependencias: PyTorch.
Módulos
  •	Ubicación: src/core/modules/
  •	Propósito: Ejecutan funciones especializadas.
  •	Módulos:
  ◦	Registro (registro.py): Registra entidades y gestiona enjambres.
  ◦	Ejecución (ejecucion.py): Orquesta la ejecución de entidades.
  ◦	Sincronización (sincronizacion.py): Limpia nodos inactivos y balancea carga.
  ◦	Auditoría (auditoria.py): Monitorea errores y micro-células débiles, generando alertas.
  •	Dependencias: PostgreSQL, Redis.
PluginManager
  •	Ubicación: src/plugins/plugin_manager.py
  •	Propósito: Carga y gestiona plugins dinámicamente (actualmente vacío).
  •	Funcionalidades:
  ◦	Escanea src/plugins/ para plugins (cargar_plugins).
  ◦	Registra canales de plugins (register_plugin_channels).
  ◦	Proporciona procesadores para canales (get_processor).
  •	Dependencias: Ninguna (directorio src/plugins/ está vacío).
OpenRouterClient
  •	Ubicación: src/utils/openrouter.py
  •	Propósito: Proporciona análisis y chat mediante OpenRouter, con fallbacks locales.
  •	Funcionalidades:
  ◦	Envía consultas a OpenRouter (query).
  ◦	Analiza datos (analyze) con respaldo local si falla.
  ◦	Maneja interacciones conversacionales (chat) con respuestas básicas si no está disponible.
  •	Dependencias: aiohttp (opcional).
Base de Datos
  •	Tablas:
  ◦	nodos: Almacena información de nodos (ID, actividad, carga, espejos).
  ◦	eventos: Registra eventos generados por entidades.
  ◦	auditoria: Guarda logs de auditoría para errores y alertas.
  •	Índices: idx_eventos_canal, idx_eventos_timestamp para consultas eficientes.
  •	Ubicación: schema.sql

Requisitos Previos
  •	Python: 3.11+
  •	Docker: Para despliegue en contenedores.
  •	PostgreSQL: Versión 15, para almacenamiento persistente.
  •	Redis: Versión 7.2, para streams.
  •	Dependencias de Python: Listadas en requirements.txt.
  •	Clave API de OpenRouter (opcional): Para análisis y chat.
  •	Sistema Operativo: Linux, macOS o Windows (con WSL para Docker en Windows).
Instala Docker:
  •	Ubuntu: sudo apt-get install docker.io docker-compose
  •	macOS: brew install docker docker-compose
  •	Windows: Descarga Docker Desktop
Instala Python:
  •	Ubuntu: sudo apt-get install python3.11 python3-pip
  •	macOS: brew install python@3.11
  •	Windows: Descarga desde python.org

Configuración
Estructura del Proyecto
corec_v4/
├── src/
│   ├── core/              # Núcleo y componentes principales
│   ├── plugins/           # Infraestructura para plugins (vacía)
│   ├── utils/             # Utilidades (logging, config, OpenRouter)
├── configs/
│   ├── core/              # Configuraciones del núcleo
│   │   ├── corec_config_corec1.json
│   │   └── secrets/       # Credenciales sensibles
├── scripts/               # Scripts de operación
├── tests/                 # Pruebas unitarias
├── docker/                # Configuraciones de Docker
├── requirements.txt       # Dependencias
├── .env                   # Variables de entorno
├── .gitignore             # Archivos ignorados
├── setup.py               # Instalación del paquete
├── schema.sql             # Esquema de la base de datos
└── main.py                # Punto de entrada
Configuración del Entorno
  1	Variables de Entorno:
  ◦	Copia el archivo de ejemplo: cp .env.example .env
  ◦	
  ◦	Edita .env: OPENROUTER_API_KEY=tu_clave_api_aquí
  ◦	ENVIRONMENT=development
  ◦	INSTANCE_ID=corec1
  ◦	
  2	Configuraciones:
  ◦	corec_config_corec1.json: {
  ◦	  "instance_id": "corec1",
  ◦	  "rol": "generica",
  ◦	  "redis_config": {
  ◦	    "host": "redis",
  ◦	    "port": 6379
  ◦	  },
  ◦	  "modulos": ["registro", "ejecucion", "sincronizacion", "auditoria"]
  ◦	}
  ◦	
  ◦	db_config.yaml: host: "postgres"
  ◦	port: 5432
  ◦	database: "corec_db"
  ◦	user: "corec_user"
  ◦	password: "secure_password"
  ◦	
  ◦	redis_config.yaml: host: "redis"
  ◦	port: 6379
  ◦	
  ◦	openrouter.yaml: enabled: true
  ◦	api_key: "tu_clave_api_aquí"
  ◦	endpoint: "https://openrouter.ai/api/v1"
  ◦	model: "nous-hermes-2"
  ◦	max_tokens: 1000
  ◦	temperature: 0.7
  ◦	
  3	Base de Datos:
  ◦	Ejecuta el esquema SQL para crear las tablas: ./scripts/init_db.sh
  ◦	
  ◦	Verifica las tablas: docker exec -it corec_v4-postgres-1 psql -U corec_user -d corec_db -c "\dt"
  ◦	
  ◦	Deberías ver nodos, eventos, y auditoria.
  4	Instala Dependencias: pip install -r requirements.txt
  5	

Ejecución
Con Docker
  1	Inicia el Sistema: ./scripts/start.sh
  2	 Esto lanza corec1, PostgreSQL y Redis.
  3	Verifica los Logs: docker logs corec_v4-corec1-1
  4	 Busca [CoreCNucleus-corec1] Inicializado.
  5	Inspecciona la Base de Datos: docker exec -it corec_v4-postgres-1 psql -U corec_user -d corec_db -c "SELECT * FROM nodos;"
  6	docker exec -it corec_v4-postgres-1 psql -U corec_user -d corec_db -c "SELECT * FROM eventos;"
  7	
  8	Detiene el Sistema: ./scripts/stop.sh
  9	
Sin Docker
  1	Inicia PostgreSQL y Redis localmente:
  ◦	PostgreSQL: sudo service postgresql start
  ◦	Redis: redis-server
  2	Ejecuta el Núcleo: python main.py
  3	
  4	Verifica los Logs: Revisa corec.log en la raíz del proyecto.

Pruebas
CoreC incluye pruebas unitarias para validar el núcleo, entidades, módulos y OpenRouter.
Ejecutar Pruebas
pytest tests/
Estructura de Pruebas
  •	tests/core/test_nucleus.py: Prueba la inicialización y funciones de CoreCNucleus.
  •	tests/core/test_celu_entidad.py: Valida el procesamiento de CeluEntidadCoreC.
  •	tests/core/test_micro_celu.py: Verifica MicroCeluEntidadCoreC y su ADN.
  •	tests/core/test_modules.py: Comprueba los módulos.
  •	tests/core/test_openrouter.py: Asegura que OpenRouter funcione con fallbacks.
  •	tests/plugins/test_plugin_manager.py: Confirma que PluginManager maneja un directorio vacío.
Ejemplo de Prueba
# tests/core/test_nucleus.py
@pytest.mark.asyncio
async def test_nucleus_inicializar():
    nucleus = CoreCNucleus(instance_id="test_corec1")
    await nucleus.inicializar()
    assert len(nucleus.modulos) == 4
    await nucleus.detener()

Monitoreo
CoreC ofrece varias formas de monitorear su estado:
Logs
  •	Ubicación: corec.log
  •	Contenido: Inicialización, errores, eventos de módulos y entidades.
  •	Ejemplo: 2025-04-17 12:00:00 [CoreCNucleus-corec1] INFO: Inicializado
  •	2025-04-17 12:00:01 [CeluEntidad-nano_test_corec1] INFO: Iniciada en canal test_canal
  •	
Base de Datos
  •	Tablas:
  ◦	nodos: Estado de entidades (ID, carga, actividad).
  ◦	eventos: Resultados de procesamiento.
  ◦	auditoria: Alertas y errores.
  •	Consulta: docker exec -it corec_v4-postgres-1 psql -U corec_user -d corec_db -c "SELECT * FROM auditoria;"
  •	
Métricas
  •	Prometheus: Medidores en src/utils/metrics.py:
  ◦	corec_celu_count: Número de CeluEntidadCoreC.
  ◦	corec_micro_count: Número de MicroCeluEntidadCoreC.
  •	Integración: Configura Prometheus y Grafana para visualizar métricas (Etapa 2).

Extensión del Sistema
CoreC está diseñado para ser extensible mediante plugins, aunque en la Etapa 1 el directorio src/plugins/ está vacío.
Estructura de un Plugin
Un plugin típico incluye:
  •	Directorio: src/plugins//
  •	plugin.json: Configuración del plugin: {
  •	  "name": "",
  •	  "version": "1.0.0",
  •	  "description": "Descripción",
  •	  "type": "processor",
  •	  "channels": ["canal1", "canal2"],
  •	  "dependencies": [],
  •	  "config_file": "configs/plugins//.yaml",
  •	  "main_class": ".processors.."
  •	}
  •	
  •	Procesador: Clase que extiende ProcesadorBase en src/plugins//processors/.
  •	Configuración: Archivo YAML en configs/plugins//.
Creando un Plugin
  1	Crea el directorio src/plugins//.
  2	Añade plugin.json con la configuración.
  3	Implementa una clase procesadora: # src/plugins//processors/.py
  4	from ....core.processors.base import ProcesadorBase
  5	from ....utils.logging import logger
  6	
  7	class NombreProcessor(ProcesadorBase):
  8	    def __init__(self, config, redis_client, db_config):
  9	        super().__init__()
  10	        self.config = config
  11	        self.logger = logger.getLogger("NombreProcessor")
  12	
  13	    async def inicializar(self, nucleus):
  14	        self.nucleus = nucleus
  15	        self.logger.info("NombreProcessor inicializado")
  16	
  17	    async def procesar(self, datos, contexto):
  18	        return {"estado": "ok", "mensaje": "Procesado"}
  19	
  20	    async def detener(self):
  21	        self.logger.info("NombreProcessor detenido")
  22	
  23	Crea la configuración en configs/plugins//.yaml: channels:
  24	  - "canal1"
  25	
  26	Instala dependencias del plugin (si las hay): pip install -r src/plugins//requirements.txt
  27	
Ejemplo de Uso
En la Etapa 2, podrías crear plugins para:
  •	CLI: Visualización interactiva con Textual.
  •	Alertas: Notificaciones externas para el canal alertas.
  
Mejores Prácticas
  •	Configuración Segura:
  ◦	Mantén las credenciales en configs/core/secrets/ fuera del control de versiones.
  ◦	Usa .env para claves API sensibles.
  •	Pruebas:
  ◦	Escribe pruebas para nuevos componentes en tests/.
  ◦	Valida fallbacks de OpenRouter con enabled: false.
  •	Monitoreo:
  ◦	Revisa corec.log regularmente.
  ◦	Configura alertas para errores en auditoria (Etapa 2).
  •	Escalabilidad:
  ◦	Ajusta max_enjambre_por_canal en ModuloRegistro para alta carga.
  ◦	Usa Kubernetes para instancias múltiples (Etapa 2).
  •	Resiliencia:
  ◦	Habilita espejos para canales críticos en canales_criticos.
  ◦	Monitorea micro-células débiles (fitness < 0.1).

Solución de Problemas
  •	Error: “FATAL: database corec_db does not exist”:
  ◦	Crea la base de datos: docker exec -it corec_v4-postgres-1 psql -U postgres -c "CREATE DATABASE corec_db;"
  ◦	
  •	Error: “Connection refused”:
  ◦	Asegúrate de que PostgreSQL y Redis estén corriendo: docker ps
  ◦	
  ◦	Verifica host en db_config.yaml (postgres) y redis_config.yaml (redis).
  •	OpenRouter no responde:
  ◦	Confirma que enabled: true y la clave API en openrouter.yaml son correctos.
  ◦	Prueba con enabled: false para usar fallbacks.
  •	Pruebas fallan:
  ◦	Revisa los logs de pytest: pytest tests/ -v
  ◦	
  ◦	Asegúrate de que la base de datos esté inicializada (./scripts/init_db.sh).
  •	Micro-células no se registran:
  ◦	Verifica max_enjambre_por_canal en ModuloRegistro.
  ◦	Inspecciona Redis streams: docker exec -it corec_v4-redis-1 redis-cli XREAD STREAMS corec_stream_corec1 0
  ◦	

Contribución
Para contribuir a CoreC:
  1	Fork el repositorio.
  2	Crea una rama: git checkout -b feature/tu-funcionalidad.
  3	Realiza cambios y escribe pruebas.
  4	Commit: git commit -m "Añade tu funcionalidad".
  5	Push: git push origin feature/tu-funcionalidad.
  6	Abre un pull request.
Sigue las convenciones de código:
  •	Usa PEP 8 para el estilo.
  •	Documenta funciones con docstrings.
  •	Incluye pruebas en tests/.

Licencia
CoreC está licenciado bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

CoreC: Un sistema vivo, donde la biología y la tecnología se fusionan para crear soluciones adaptativas y escalables.
