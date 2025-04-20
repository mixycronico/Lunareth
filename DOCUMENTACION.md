# Documentación Técnica de CoreC

## Índice

1. [Introducción](#introducción)  
2. [Arquitectura](#arquitectura)  
   2.1. [Entidades](#entidades)  
   2.2. [Bloques Simbióticos](#bloques-simbióticos)  
   2.3. [Núcleo (CoreCNucleus)](#núcleo-corecnucleus)  
   2.4. [Módulos](#módulos)  
   2.5. [Comunicación](#comunicación)  
   2.6. [Almacenamiento](#almacenamiento)  
3. [Requisitos](#requisitos)  
4. [Instalación](#instalación)  
   4.1. [Entorno Linux](#entorno-linux)  
   4.2. [Dependencias Python](#dependencias-python)  
   4.3. [Despliegue con Docker](#despliegue-con-docker)  
5. [Uso](#uso)  
   5.1. [Iniciar CoreC](#iniciar-corec)  
   5.2. [Monitoreo y Logs](#monitoreo-y-logs)  
   5.3. [Ejecutar Pruebas](#ejecutar-pruebas)  
6. [Configuración](#configuración)  
   6.1. [corec_config.json](#corec_configjson)  
7. [Escalabilidad y Rendimiento](#escalabilidad-y-rendimiento)  
   7.1. [Multi‑Nodo](#multi‑nodo)  
   7.2. [Optimización de Recursos](#optimización-de-recursos)  
8. [Solución de Problemas](#solución-de-problemas)  
9. [Futuras Mejoras](#futuras-mejoras)  
10. [Desarrolladores](#desarrolladores)  

---

## 1. Introducción

**CoreC** es un framework bioinspirado, diseñado para procesar hasta **1 000 000** de entidades ultraligeras (≈ 1 KB c/u) dentro de **1 GB** de RAM. Su arquitectura modular y distribuida garantiza:

- **Alta concurrencia**: Asyncio + Celery  
- **Baja latencia**: < 20 ms por bloque  
- **Alta disponibilidad**: > 99.99 % uptime  
- **Escalabilidad**: Redis Cluster + PostgreSQL particionado  

---

## 2. Arquitectura

### 2.1. Entidades

- **MicroCeluEntidadCoreC**  
  - Tupla `(id: str, canal: int, función: Callable[[], Awaitable[Dict]], activo: bool)`  
  - Memoria ≈ 1 KB  
  - Ejecuta una coroutine sin parámetros, devuelve `{"valor": float}`  

- **CeluEntidadCoreC**  
  - Tupla `(id: str, canal: int, procesador: Callable[[Dict], Awaitable[Dict]], activo: bool)`  
  - Memoria ≈ 1.5 KB  
  - Procesa un payload entrante, devuelve `{"valor": float}`  

Las entidades serializan/deserializan con `struct.pack("!Ibf?", …)` (ID:uint32, canal:uint8, valor:float32, estado:bool).

### 2.2. Bloques Simbióticos

- **Tamaño estándar**: ~1 MB (≈ 1000 entidades)  
- **Tamaño inteligente**: ~5 MB (≈ 2000 entidades + modelo ligero)  

**Flujo de procesamiento (O(n))**  
1. **Selección**: `n = ceil(len(entidades) * carga)`  
2. **Ejecución**: `await asyncio.gather(...)`  
3. **Deserialización**: `struct.unpack`  
4. **Cálculo de fitness**: desviación estándar y promedios  
5. **Autorreparación**: recarga de funciones en errores > umbral  
6. **Persistencia**: compresión Zstandard + `INSERT` en PostgreSQL  

> **Rendimiento típico**: 5–10 ms por bloque de 1000 entidades.

### 2.3. Núcleo (CoreCNucleus)

Responsable de:

1. **Carga de configuración** (`corec_config.json`).  
2. **Infraestructura**:  
   - `init_postgresql(db_config)` → crea tablas/particiones.  
   - `init_redis(redis_config)` → instancia `aioredis.Redis`.  
3. **Registro dinámico**:  
   - Módulos en `corec/modules/`  
   - Plugins en `plugins/…/main.py`  
4. **Loop principal**:  
   ```python
   tasks = [m.ejecutar() for m in self.modules.values()]
   await asyncio.gather(*tasks)

    5.	Alertas: publicación centralizada vía Celery.
    6.	Cierre limpio: detiene módulos y cierra Redis.

2.4. Módulos

Cada módulo hereda de ModuloBase e implementa:
    •	async def inicializar(self, nucleus): …
    •	async def ejecutar(self): …
    •	async def detener(self): …

Módulo	Función
registro	Inicializa y registra bloques según configuración.
ejecucion	Encola procesamiento de bloques con Celery.
sincronizacion	Fusiona/adapta bloques en función de carga y fitness.
auditoria	Detecta anomalías (IsolationForest) y dispara alertas.

2.5. Comunicación
    •	Formato: "!Ibf?" (big‑endian)
    •	Transport: Redis Streams (XADD / XREAD)
    •	Canales:
    1.	Genérico
    2.	Seguridad (alta prioridad)
    3.	IA / auditoría
    4.	Alertas críticas

Streams configurables con maxlen y TTL para idempotencia.

2.6. Almacenamiento
    •	PostgreSQL particionado por rango de timestamp.
    •	Esquema bloques:

CREATE TABLE bloques (
  id TEXT,
  canal INT,
  num_entidades INT,
  fitness REAL,
  timestamp DOUBLE PRECISION,
  instance_id TEXT
) PARTITION BY RANGE (timestamp);


    •	Optimización: índices compuestos (canal, timestamp DESC).
    •	Compresión: Zstandard nivel 3 para JSON.

⸻

3. Requisitos
    •	Hardware:
    •	Mínimo: 2 GB RAM, 2 vCPU, 10 GB disco
    •	Recomendado: 4 GB RAM, 4 vCPU, 20 GB disco
    •	Software:
    •	Python ≥ 3.9
    •	Redis ≥ 7.0
    •	PostgreSQL ≥ 13
    •	Linux (x86_64 / ARM64)

⸻

4. Instalación

4.1. Entorno Linux

sudo apt update
sudo apt install -y python3-pip python3-dev libpq-dev gcc g++
sudo apt install -y redis-server postgresql postgresql-contrib

4.2. Dependencias Python

pip install --upgrade pip
pip install -r requirements.txt

4.3. Despliegue con Docker

# docker-compose.yml
version: '3.8'
services:
  redis:
    image: redis:7
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: corec_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_password
  corec:
    build: .
    volumes:
      - ./:/app
    working_dir: /app
    command: ["./run.sh"]
    depends_on: [redis, postgres]



⸻

5. Uso

5.1. Iniciar CoreC

./run.sh
# Arranca:
#  • CoreC (bootstrap → CoreCNucleus)
#  • Celery workers

5.2. Monitoreo y Logs
    •	Local: tail -f logs/corec.log
    •	Docker: docker-compose logs -f corec

Buscar etiquetas clave:
    •	[CoreCNucleus] Inicializado
    •	[ModuloRegistro] Bloque registrado

5.3. Ejecutar Pruebas

pytest -q

Cobertura de:
    •	Serialización
    •	Entidades
    •	Bloques
    •	Módulos
    •	Núcleo

⸻

6. Configuración

6.1. corec_config.json

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
      "entidades_por_bloque": 2000,
      "plugin": "procesador_inteligente"
    }
  ],
  "plugins": [
    { "name": "codex", "enabled": true, "config_path": "plugins/codex/config.json" },
    { "name": "crypto_trading", "enabled": true, "config_path": "plugins/crypto_trading/config.json" },
    { "name": "interface_system", "enabled": true, "config_path": "plugins/interface_system/config.json" }
  ]
}



⸻

7. Escalabilidad y Rendimiento

7.1. Multi‑Nodo
    •	Redis Cluster en puertos 7000–7005
    •	PostgreSQL particionado por rango y canal
    •	Celery distribuido en múltiples workers

7.2. Optimización de Recursos
    •	Memoria
    •	Reposo: 100–150 MB
    •	Pico: ~1 GB (1 000 000 entidades)
    •	CPU
    •	Concurrencia cooperativa: 0.3–0.5 vCPU para ~1000 t/s
    •	I/O
    •	Redis Streams: 0.2–0.3 MB/s
    •	PostgreSQL: 1–10 ops/s por bloque

⸻

8. Solución de Problemas
    •	ImportError: corec.modules
    •	Asegurar corec/modules/__init__.py existe.
    •	Redis ConnectionError

redis-cli -h host -p port -a password ping


    •	PostgreSQL OperationalError

sudo systemctl status postgresql


    •	Bloques no aparecen
    •	Revisar sección bloques en corec_config.json.

⸻

9. Futuras Mejoras
    •	CLI interactivo para gestión en runtime
    •	Dashboard Grafana/Prometheus con métricas en tiempo real
    •	Evolución orgánica: mutación dinámica de funciones de entidades
    •	WebAssembly para procesadores personalizados

⸻

10. Desarrolladores
    •	Moises Alvarenga
    •	Luna

Documento generado para ofrecer una visión técnica y detallada de CoreC, con todos sus componentes y flujos de trabajo.