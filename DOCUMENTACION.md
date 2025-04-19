Documentación Completa de CoreC
Índice
  1	Introducción
  2	Arquitectura
  ◦	Entidades
  ◦	Bloques Simbióticos
  ◦	Núcleo
  ◦	Módulos
  ◦	Plugins
  ◦	Comunicación
  ◦	Almacenamiento
  3	Requisitos
  4	Instalación
  ◦	Entorno Linux
  ◦	Configuración de Dependencias
  ◦	Despliegue con Docker
  5	Uso
  ◦	Iniciar CoreC
  ◦	Monitoreo
  ◦	Pruebas
  6	Configuración
  ◦	corec_config.json
  ◦	plugins_config.json
  7	Desarrollo de Plugins
  ◦	Estructura de un Plugin
  ◦	Ejemplo: Plugin de Alerta
  8	Escalabilidad y Rendimiento
  ◦	Multi-Nodo
  ◦	Optimización de Recursos
  9	Solución de Problemas
  10	Futuras Mejoras
  11	Alineación con la Visión

1. Introducción
CoreC es un ecosistema digital bioinspirado diseñado para procesar y coordinar millones de entidades ultraligeras (~1 KB) en bloques simbióticos (~1 MB, ~5 MB para bloques inteligentes) dentro de un límite de 1 GB de RAM para ~1,000,000 entidades. Inspirado en la eficiencia de GNOME 2, CoreC es plug-and-play, escalable multi-nodo, y ofrece un desempeño legendario (>99.99% uptime, ~10-20 ms latencia). Su arquitectura modular permite extender funcionalidades mediante plugins sin modificar el núcleo ni la configuración base (corec_config.json).
Objetivos
  •	Eficiencia: Operar con recursos mínimos (~100-150 MB en reposo, ~0.9-1 GB en carga alta).
  •	Modularidad: Plugins plug-and-play en subfolders dentro de plugins (por ejemplo, plugins/alerts).
  •	Escalabilidad: Soporte multi-nodo con Redis Cluster y PostgreSQL particionado.
  •	Simplicidad: Configuración y despliegue intuitivos, como GNOME 2.
Casos de Uso
  •	Procesamiento distribuido de sensores IoT.
  •	Análisis en tiempo real (por ejemplo, detección de anomalías, trading).
  •	Simulaciones bioinspiradas (por ejemplo, redes neuronales ligeras).
  •	Sistemas embebidos (por ejemplo, Raspberry Pi 5).

2. Arquitectura
CoreC está estructurado como un organismo digital, con entidades (“almas”) formando bloques simbióticos coordinados por un núcleo, módulos, y plugins.
2.1 Entidades
  •	Descripción: Unidades mínimas de procesamiento (~0.8-1.2 KB), como células en un organismo.
  •	Tipos:
  ◦	MicroCeluEntidadCoreC: Ejecuta cálculos simples (por ejemplo, random.random()).
  ▪	Estructura: Tupla (id: str, canal: int, funcion: Callable, activo: bool).
  ▪	RAM: ~0.8-1.2 KB.
  ▪	Funciones: Cálculo, filtrado dinámico (umbral ajustado por carga, desviación, errores), fusión (herencia de función), autoreparación (bit de estado, reinicio tras 2 fallos).
  ◦	CeluEntidadCoreC: Ejecuta procesadores avanzados (por ejemplo, ProcesadorSensor, ProcesadorFiltro).
  ▪	Estructura: Tupla (id: str, canal: int, procesador: Callable, activo: bool).
  ▪	RAM: ~1-1.5 KB.
  ▪	Funciones: Procesamiento complejo, filtrado, integración con plugins.
  •	Total: ~1,000,000 entidades (~980,000 en bloques de 1 MB, ~20,000 en bloques de 5 MB).
2.2 Bloques Simbióticos
  •	Descripción: Grupos de ~1000 entidades (~1 MB) o ~2000 entidades con red neuronal (~5 MB), como tejidos vivos que coordinan tareas.
  •	Tipos:
  ◦	Estándar: ~1 MB, ~1000 entidades, para cálculo, fusión, reparación, escritura agregada en PostgreSQL.
  ◦	Inteligente: ~5 MB, ~2000 entidades + red neuronal (~1 MB), para análisis avanzado (por ejemplo, predicciones).
  •	Funciones:
  ◦	Cálculo distribuido: Agrega resultados de entidades (por ejemplo, promedio de valores).
  ◦	Fusión: Combina bloques débiles (fitness < 0.2) con bloques fuertes (fitness > 0.5).
  ◦	Reparación: Regenera entidades defectuosas (>5% errores), heredando funciones.
  ◦	Escritura: Almacena datos agregados en PostgreSQL (~1-10 ops/s).
  •	RAM: ~980 MB para 980 bloques estándar + 50 MB para 10 bloques inteligentes.
2.3 Núcleo
  •	Archivo: corec/nucleus.py
  •	Descripción: El CoreCNucleus es el cerebro que coordina entidades, bloques, módulos, y plugins.
  •	Funciones:
  ◦	Carga dinámica de módulos (corec/modules) y plugins (plugins/*).
  ◦	Inicializa Redis (broker, caché, streams) y PostgreSQL (corec_db).
  ◦	Gestiona tareas Celery (~1000 tareas/s).
  ◦	Publica alertas críticas (por ejemplo, anomalías).
  •	RAM: ~50 MB.
2.4 Módulos
  •	Ubicación: corec/modules
  •	Descripción: Componentes esenciales del núcleo, implementados como clases (ModuloBase).
  •	Módulos:
  ◦	registro.py: Gestiona bloques simbióticos, registra entidades.
  ◦	ejecucion.py: Ejecuta tareas distribuidas con Celery.
  ◦	sincronización.py: Fusiona y adapta bloques según carga y fitness.
  ◦	auditoria.py: Detecta anomalías en bloques (usando IsolationForest).
  •	RAM: ~10-20 MB por módulo (~40-80 MB total).
2.5 Plugins
  •	Ubicación: plugins/* (cada plugin en su subfolder, por ejemplo, plugins/alerts).
  •	Descripción: Extensiones opcionales cargadas dinámicamente por el PluginManager sin modificar corec_config.json.
  •	Características:
  ◦	Plug-and-play: Copiar un folder a plugins y reiniciar CoreC.
  ◦	Configuración: plugins//config.json para parámetros propios (por ejemplo, bases de datos como trading_db).
  ◦	Comunicación: Usa protocolo binario (!Ibf?, ~9 bytes) vía Redis streams.
  •	RAM: ~10-20 KB por plugin (sin base propia), ~50-100 KB con base propia.
2.6 Comunicación
  •	Protocolo: Binario (!Ibf?: ID uint32, canal uint8, valor float32, estado bool, ~9 bytes).
  •	Mecanismo: Redis streams (~0.2-0.3 MB/s, ~50-100 MB RAM).
  •	Flujo:
  ◦	Entidades envían mensajes binarios a bloques.
  ◦	Bloques coordinan con otros bloques o plugins vía Redis.
  ◦	Plugins acceden a streams para tareas específicas (por ejemplo, alertas).
  •	Prioridad: Canales críticos (2: seguridad, 3: IA, 5: alertas) usan colas rápidas.
2.7 Almacenamiento
  •	Base de datos: PostgreSQL (corec_db para el núcleo, bases propias para plugins).
  •	Tablas:
  ◦	bloques: Almacena estado de bloques (id, canal, num_entidades, fitness, timestamp, instance_id).
  ◦	Particionada por rango de tiempo (bloques_2025_04).
  •	Escrituras: ~1-10 ops/s por bloque, agregadas (~95% menos que por entidad).
  •	RAM: ~50 MB.
  •	Disco: ~100-200 MB.

3. Requisitos
Hardware
  •	Mínimo: 2 GB RAM, 2 núcleos, 10 GB disco.
  •	Recomendado: 4 GB RAM, 4 núcleos, 20 GB disco (por ejemplo, Raspberry Pi 5, servidor pequeño).
Software
  •	Sistema operativo: Linux (Ubuntu 22.04, Debian 11, Raspberry Pi OS 64-bit).
  •	Dependencias:
  ◦	Python 3.9
  ◦	Docker, Docker Compose
  ◦	Redis 7.0
  ◦	PostgreSQL 13
  ◦	Bibliotecas Python: celery, redis, aioredis, psycopg2-binary, zstd, scikit-learn, torch, jq
Red
  •	Puertos: 6379 (Redis), 5432 (PostgreSQL), 8000 (métricas).

4. Instalación
4.1 Entorno Linux
  1	Actualiza el sistema: sudo apt update
  2	
  3	Instala dependencias del sistema: sudo apt install -y python3.9 python3-pip python3-dev libpq-dev gcc g++ docker.io docker-compose jq
  4	sudo systemctl enable docker
  5	sudo systemctl start docker
  6	sudo usermod -aG docker $USER
  7	
  8	Descarga el código de CoreC (por ejemplo, desde tu servidor HelloAmigo): git clone  corec
  9	cd corec
  10	
4.2 Configuración de Dependencias
  1	Instala Redis: sudo apt install -y redis-server
  2	sudo sed -i 's/# requirepass .*/requirepass secure_password/' /etc/redis/redis.conf
  3	sudo systemctl restart redis
  4	
  5	Instala PostgreSQL: sudo apt install -y postgresql postgresql-contrib
  6	sudo -u postgres psql -c "CREATE DATABASE corec_db;"
  7	sudo -u postgres psql -c "ALTER USER postgres WITH PASSWORD 'your_password';"
  8	
  9	Instala dependencias de Python: pip install -r requirements.txt
  10	
  11	Ejecuta el script de configuración: chmod +x setup.sh
  12	./setup.sh
  13	
4.3 Despliegue con Docker
  1	Actualiza configs/corec_config.json con credenciales reales: {
  2	    "instance_id": "corec1",
  3	    "db_config": {
  4	        "dbname": "corec_db",
  5	        "user": "postgres",
  6	        "password": "your_real_password",
  7	        "host": "localhost",
  8	        "port": "5432"
  9	    },
  10	    "redis_config": {
  11	        "host": "localhost",
  12	        "port": 6379,
  13	        "username": "",
  14	        "password": "your_real_redis_password"
  15	    },
  16	    ...
  17	}
  18	
  19	Inicia CoreC: ./run.sh
  20	
  21	Verifica los logs: docker-compose logs corec
  22	

5. Uso
5.1 Iniciar CoreC
  1	Ejecuta: ./run.sh
  2	
  ◦	Inicia los servicios corec, celery, redis, y postgres.
  3	Inicia workers de Celery (en otra terminal): celery -A corec.core.celery_app worker --loglevel=info
  4	
5.2 Monitoreo
  •	Logs: docker-compose logs corec
  •	
  ◦	Busca mensajes como [CoreCNucleus-corec1] Inicializado, [ModuloRegistro] Bloque registrado.
  •	Prometheus (opcional):
  ◦	Configura Prometheus y Grafana: docker run -d -p 9090:9090 prom/prometheus
  ◦	docker run -d -p 3000:3000 grafana/grafana
  ◦	
  ◦	Edita monitoring/prometheus.yml para métricas de Celery, Redis, y PostgreSQL.
5.3 Pruebas
  •	Ejecuta: python -m tests.test_corec
  •	
  •	Valida:
  ◦	RAM: ~0.9-1 GB para ~1,000,000 entidades.
  ◦	CPU: ~0.3-0.5 núcleo.
  ◦	Red: ~0.2-0.3 MB/s.
  ◦	Escrituras: ~1-10 ops/s por bloque.

6. Configuración
6.1 corec_config.json
Define el núcleo, bloques, y conexiones.
{
    "instance_id": "corec1",
    "db_config": {
        "dbname": "corec_db",
        "user": "postgres",
        "password": "your_real_password",
        "host": "localhost",
        "port": "5432"
    },
    "redis_config": {
        "host": "localhost",
        "port": 6379,
        "username": "",
        "password": "your_real_redis_password"
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
            "entidades_por_bloque": 2000,
            "plugin": "procesador_inteligente"
        }
    ],
    "modules": ["registro", "ejecucion", "sincronización", "auditoria"],
    "rol": "generica"
}
  •	instance_id: Identificador único del nodo.
  •	db_config: Conexión a PostgreSQL (corec_db).
  •	redis_config: Conexión a Redis (broker, streams).
  •	bloques: Define bloques iniciales (estándar e inteligentes).
  •	modules: Lista de módulos activos.
  •	rol: Función del nodo (por ejemplo, “generica”, “sensor”, “analisis”).
6.2 plugins_config.json
Configuraciones globales opcionales para plugins (complementa plugins//config.json).
{
    "alerts": {
        "enabled": true,
        "bloque": "alerts_notificaciones",
        "canal": 5
    }
}

7. Desarrollo de Plugins
7.1 Estructura de un Plugin
  •	Ubicación: plugins/ (por ejemplo, plugins/alerts).
  •	Archivos:
  ◦	main.py: Punto de entrada con una clase que implementa inicializar(nucleus, config).
  ◦	config.json (opcional): Configuración propia (por ejemplo, base de datos, canal).
  •	Carga: El PluginManager en nucleus.py escanea plugins/*/ y carga main.py dinámicamente.
  •	Plug-and-play: Copiar un folder a plugins y reiniciar CoreC activa el plugin sin modificar corec_config.json.
7.2 Ejemplo: Plugin de Alerta
El plugin Alerts detecta valores extremos en bloques simbióticos y envía notificaciones a Redis streams.
`plugins/alerts/main.py`
from corec.core import logging, asyncio, enviar_mensaje_redis, serializar_mensaje, deserializar_mensaje, random
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad
from typing import Dict, Any

class AlertsPlugin:
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.logger = logging.getLogger("AlertsPlugin")
        self.bloque_id = config.get("bloque", "alerts_notificaciones")
        self.canal = config.get("canal", 5)  # Canal 5 para alertas
        self.umbral_alerta = config.get("umbral_alerta", 0.9)

    async def inicializar(self):
        """Inicializa el plugin y registra su bloque."""
        entidades = []
        for i in range(1000):  # Bloque estándar de 1 MB
            async def funcion():
                return {"valor": random.random()}
            entidades.append(crear_entidad(f"m{i}", self.canal, funcion))
        bloque = BloqueSimbiotico(self.bloque_id, self.canal, entidades, max_size=1024, nucleus=self.nucleus)
        modulo_registro = self.nucleus.modulos.get("registro")
        if modulo_registro:
            modulo_registro.bloques[self.bloque_id] = bloque
            self.logger.info(f"Bloque alertas {self.bloque_id} registrado con 1000 entidades")

    async def procesar(self, datos: Dict[str, Any], carga: float):
        """Procesa valores y envía alertas si superan el umbral."""
        bloque = self.nucleus.modulos.get("registro").bloques.get(self.bloque_id)
        resultados = await bloque.procesar(carga)
        valores = [r["valor"] for r in resultados["mensajes"] if r.get("activo")]
        max_valor = max(valores) if valores else 0.0
        if max_valor > self.umbral_alerta:
            mensaje = await serializar_mensaje(0, self.canal, max_valor, True)
            await enviar_mensaje_redis(self.nucleus.redis_client, "alertas", mensaje, prioridad="critical")
            self.logger.info(f"Alerta enviada: valor {max_valor} supera umbral {self.umbral_alerta}")
        return {"alertas": valores}

def inicializar(nucleus, config):
    """Punto de entrada del plugin."""
    plugin = AlertsPlugin(nucleus, config)
    nucleus.registrar_plugin("alerts", plugin)
`plugins/alerts/config.json`
{
    "bloque": "alerts_notificaciones",
    "enabled": true,
    "canal": 5,
    "umbral_alerta": 0.9
}
Desglose del Plugin de Alerta
  •	Propósito: Monitorea valores generados por entidades en un bloque simbiótico (~1 MB, ~1000 entidades) y envía alertas si algún valor supera un umbral (0.9).
  •	Estructura:
  ◦	main.py: Define AlertsPlugin con:
  ▪	__init__: Configura el bloque, canal, y umbral.
  ▪	inicializar: Crea un bloque con 1000 entidades y lo registra.
  ▪	procesar: Analiza valores y envía alertas críticas a Redis streams.
  ▪	inicializar: Punto de entrada para el PluginManager.
  ◦	config.json: Especifica el bloque (alerts_notificaciones), canal (5), y umbral (0.9).
  •	Uso:
  ◦	Copia plugins/alerts/ a tu servidor HelloAmigo.
  ◦	Reinicia CoreC: ./run.sh
  ◦	
  ◦	Verifica logs: docker-compose logs corec
  ◦	
  ▪	Busca [AlertsPlugin] Alerta enviada si un valor supera 0.9.
  •	Consumo:
  ◦	RAM: ~10-20 KB (sin base propia).
  ◦	CPU: ~0.001-0.01% núcleo por ejecución.
  ◦	Red: ~0.01 MB/s (~10 mensajes/s en alertas críticas).

8. Escalabilidad y Rendimiento
8.1 Multi-Nodo
  •	Redis Cluster:
  ◦	Configura múltiples nodos Redis: docker run -d -p 7000-7005:7000-7005 redis:7.0 --cluster-enabled yes
  ◦	
  ◦	Actualiza redis_config en corec_config.json: {
  ◦	    "redis_config": {
  ◦	        "host": "redis-cluster",
  ◦	        "port": 7000,
  ◦	        "username": "",
  ◦	        "password": "secure_password"
  ◦	    }
  ◦	}
  ◦	
  •	PostgreSQL Particionado:
  ◦	Usa particiones por tiempo (bloques_2025_04) y canal para escalar escrituras.
  ◦	Configura réplicas para nodos adicionales: docker run -d -p 5433:5432 postgres:13 --replication
  ◦	
8.2 Optimización de Recursos
  •	RAM:
  ◦	Reposo: ~100-150 MB (1 bloque activo).
  ◦	Carga media: ~160-260 MB (10 bloques).
  ◦	Carga alta: ~0.9-1 GB (~980 bloques estándar + 10 inteligentes).
  •	CPU: ~0.3-0.5 núcleo para ~1000 tareas/s.
  •	Disco: ~50-100 MB (particiones optimizadas).
  •	Red: ~0.2-0.3 MB/s (~1000 mensajes/s).
  •	Técnicas:
  ◦	Carga dinámica: Entidades inactivas liberan RAM (~0.1 KB por entidad).
  ◦	Escrituras agregadas: ~95% menos ops/s que por entidad.
  ◦	Compresión zstd: ~0.1 KB por mensaje.

9. Solución de Problemas
  •	ImportError: No module named ‘corec.modules’:
  ◦	Verifica que corec/modules/__init__.py existe (vacío).
  ◦	Solución: touch corec/modules/__init__.py
  ◦	
  •	Redis ConnectionError:
  ◦	Confirma que Redis está activo: redis-cli -h localhost -p 6379 -a secure_password ping
  ◦	
  ◦	Actualiza redis_config en corec_config.json.
  •	PostgreSQL OperationalError:
  ◦	Verifica credenciales en corec_config.json.
  ◦	Asegúrate de que PostgreSQL está corriendo: sudo systemctl status postgresql
  ◦	
  •	Plugin no carga:
  ◦	Confirma que plugins//main.py tiene inicializar(nucleus, config).
  ◦	Verifica config.json (por ejemplo, enabled: true).
  ◦	Revisa logs: docker-compose logs corec
  ◦	
  •	RAM excede 1 GB:
  ◦	Reduce entidades activas (~1,000,000 → ~500,000).
  ◦	Limita workers de Celery: # corec/core.py
  ◦	worker_concurrency = 2
  ◦	

10. Futuras Mejoras
  •	Expansión: ~3,333,333 entidades con ~2 GB RAM y compresión binaria (~0.5 KB por entidad).
  •	CLI: Gestión de plugins (corec plugin add ).
  •	Visualización: Panel de Grafana para bloques, entidades, y alertas.
  •	Raspberry Pi: Imagen Docker optimizada (python:3.9-slim, workers = 2).
  •	Modo Orgánico: Sinapsis artificial y mutación de entidades (por ejemplo, evolución de funciones).

11. Alineación con la Visión
  •	Eficiencia (9 de abril de 2025): Entidades como “almas” (~1 KB) forman bloques simbióticos (~1 MB, ~5 MB), dentro de 1 GB, emulando la ligereza de GNOME 2.
  •	Escalabilidad (13 de abril de 2025): ~1000 bloques distribuidos con Celery y Redis soportan multi-nodo.
  •	Simplicidad (14 de abril de 2025): Plug-and-play con plugins en subfolders (plugins/alerts), sin modificar corec_config.json.
  •	Desempeño legendario (17 de abril de 2025): Latencia baja (~10-20 ms), autoreparación, y estabilidad (>99.99% uptime).

