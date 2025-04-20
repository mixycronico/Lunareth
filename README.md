# CoreC

> **Un ecosistema bioinspirado** para el procesamiento masivo de **entidades ultraligeras** y coordinación en **bloques simbióticos**, optimizado para alta concurrencia y baja latencia.

---

## 🌟 Características clave

- **Entidades ligeras** (~1 KB cada una)  
- **Bloques simbióticos** de 1 MB (≈ 1000 entidades) o 5 MB (≈ 2000 entidades + IA)  
- **Concurrencia asíncrona** con asyncio  
- **Tareas distribuidas** mediante Celery + Redis Streams  
- **Almacenamiento particionado** en PostgreSQL con Zstd  
- **Modularidad total**: núcleo y módulos desacoplados  

---

## 🏛 Arquitectura en un vistazo

┌─────────────────────────────────────────────┐
│                 Bootstrap                  │
│     (corec/bootstrap.py → run.sh)          │
└───────────────┬─────────────────────────────┘
│
▼
CoreCNucleus (corec/nucleus.py)
├─ Carga de configuración JSON
├─ init_postgresql(db_config)
├─ init_redis(redis_config)
├─ Registro dinámico de módulos
└─ Loop principal: asyncio.gather(modulos)
│
▼
┌────────────────────────────────────┐
│           Módulos CoreC           │
│ (corec/modules/*.py heredan de)   │
│             ModuloBase            │
└────────────────────────────────────┘

- **Entidades** (`corec/entities.py`)  
  - `MicroCeluEntidadCoreC`: tuplas `(id, canal, func, activo)` → cálculos rápidos  
  - `CeluEntidadCoreC`: tuplas `(id, canal, procesador, activo)` → procesamiento con datos  

- **Bloques** (`corec/blocks.py`)  
  - Ciclo:  
    1. procesar entidades (`asyncio.gather`),  
    2. ajustar umbral (desviación + carga + errores),  
    3. autoreparación si >5 % fallos,  
    4. escribir resultados en PostgreSQL (Zstd + INSERT ON CONFLICT)  

- **Comunicación**  
  - Formato binario: `!Ibf?` (4 bytes ID, 1 byte canal, 4 bytes float, 1 byte bool)  
  - Redis Streams para mensajería de alta velocidad  

- **Almacenamiento**  
  - Tabla `bloques` particionada por rango de `timestamp`  
  - Índices `(canal, timestamp DESC)`  
  - Compresión Zstd nivel 3 para payloads JSON  

---

## 🚀 Instalación rápida

1. **Clona** el repositorio  
   ```bash
   git clone https://github.com/tu_usuario/corec.git
   cd corec

   2.	Instala dependencias Python

pip install -r requirements.txt


   3.	Configura configs/corec_config.json

{
  "instance_id": "corec1",
  "db_config": {
    "dbname": "corec_db",
    "user": "postgres",
    "password": "YOUR_PG_PASSWORD",
    "host": "localhost",
    "port": 5432
  },
  "redis_config": {
    "host": "localhost",
    "port": 6379,
    "username": "corec_user",
    "password": "YOUR_REDIS_PASSWORD"
  },
  "bloques": [
    { "id": "sensor_swarm", "canal": 1, "entidades": 980000, "max_size_mb": 1 },
    { "id": "ia_analysis",  "canal": 3, "entidades": 20000,  "max_size_mb": 5 }
  ]
}


   4.	Arranca CoreC

bash run.sh


   5.	Inicia workers de Celery (otra terminal)

celery -A corec.core.celery_app worker --loglevel=info


   6.	(Opcional) Docker Compose

docker-compose up -d



⸻

🧪 Pruebas y Calidad
   •	Ejecuta tests unitarios con pytest:

pytest -q


   •	Estilo y tipado:

flake8 corec/  
mypy corec/



⸻

🔧 Uso y Monitoreo
   •	Logs:
   •	Local: tail -f logs/corec.log
   •	Docker: docker-compose logs -f corec
   •	Indicadores clave:
   •	[CoreCNucleus] Inicializado
   •	[ModuloRegistro] Bloque registrado
   •	Métricas: integra con Prometheus + Grafana para latencia, throughput y uso de memoria

⸻

⚙️ Configuración avanzada
   •	Multi‑Nodo: apuntar redis_config.host a un Redis Cluster y usar réplicas PostgreSQL
   •	Variables de entorno: reemplaza credenciales sensibles (POSTGRES_PASSWORD, REDIS_PASSWORD)
   •	Ajustes de rendimiento:
   •	reduce worker_concurrency en celery_app.conf
   •	ajusta particiones de PostgreSQL para rangos de tiempo más pequeños

⸻

🔮 Futuras mejoras
   •	CLI interactivo para gestión en caliente
   •	Evolución orgánica de entidades (mutación dinámica)
   •	WebAssembly para procesadores custom ultraligeros
   •	Dashboard integrado con métricas de bloques y fitness

⸻

👥 Desarrolladores

Moises Alvarenga & Luna
CoreC © 2025 — Todos los derechos reservados

