# CommSystem

> **Un asistente conversacional vivo** para CoreC: chat real con memoria, comandos inteligentes y generación de plugins on‑demand.

---

## 📖 Índice

1. [Descripción](#descripción)  
2. [Requisitos](#requisitos)  
3. [Estructura de carpetas](#estructura-de-carpetas)  
4. [Instalación](#instalación)  
5. [Configuración](#configuración)  
6. [Cómo funciona](#cómo-funciona)  
   - [Arrancar el plugin](#arrancar-el-plugin)  
   - [Enviar comandos](#enviar-comandos)  
   - [Ejemplos de uso](#ejemplos-de-uso)  
7. [API de Redis Streams](#api-de-redis-streams)  
8. [Extensiones futuras](#extensiones-futuras)  
9. [Autores](#autores)  

---

## 1. Descripción

**CommSystem** es un plugin para CoreC que:

- **Responde por texto** usando el modelo `google/gemini-2.0-flash-lite-001` vía OpenRouter.  
- **Recuerda** tus últimas interacciones (hasta 50 pares pregunta‑respuesta) en Redis y en disco.  
- **Despliega** un bucle asíncrono que lee comandos de Redis Streams y publica respuestas automáticas.  
- **Puede generar** nuevos plugins on‑demand (scaffold) y recargarlos en caliente.  

Ideal para sentarte frente al PC y pedirle a CoreC acciones como:

> “Crea un plugin que monitorice sensores IoT”  
> “¿Qué módulos y plugins están activos?”  
> “Muéstrame el estado de los bloques”  

…y obtener respuestas reales, con contexto y sin frases pregrabadas.

---

## 2. Requisitos

- **CoreC** vX.Y con soporte de plugins.  
- **Python 3.9+**  
- Cuenta y API‑Key en **OpenRouter.ai**  
- Redis 6+ (Streams)  
- Bibliotecas en `requirements.txt` del plugin

---

## 3. Estructura de carpetas

plugins/comm_system/
├── init.py
├── main.py
├── config.json
├── requirements.txt
└── processors/
├── init.py
├── memory.py       # Memoria conversacional
├── ai_chat.py      # Chat real con OpenRouter
└── manager.py      # Loop de comandos y despacho
└── utils/
├── init.py
└── helpers.py      # I/O asíncrono y utilidades
└── memory.json        # Respaldo en disco (se crea al usar)

---

## 4. Instalación

1. **Clona** o copia la carpeta en tu repositorio CoreC:  
   ```bash
   cp -R comm_system plugins/comm_system

  2.	Instala las dependencias del plugin:

pip install -r plugins/comm_system/requirements.txt


  3.	Exporta tu API‑Key de OpenRouter:

export OPENROUTER_API_KEY="tu_api_key_aquí"


  4.	Reinicia CoreC para que cargue el nuevo plugin:

bash run.sh



⸻

5. Configuración

Edita plugins/comm_system/config.json con tus parámetros:

{
  "comm_system": {
    "stream_in":           "corec_commands",
    "stream_out":          "corec_responses",
    "openrouter_api_key":  "${OPENROUTER_API_KEY}",
    "openrouter_api_base": "https://openrouter.ai/api/v1",
    "openrouter_model":    "google/gemini-2.0-flash-lite-001",
    "max_tokens":          200,
    "temperature":         0.7,
    "memory_ttl":          3600,
    "memory_file":         "plugins/comm_system/memory.json"
  }
}

  •	stream_in / stream_out: nombres de Redis Streams.
  •	openrouter_*: credenciales y modelo.
  •	memory_ttl: duración en segundos de la memoria en Redis.
  •	memory_file: ruta de respaldo en disco.

⸻

6. Cómo funciona

Arrancar el plugin

Al iniciar CoreC, CommSystemPlugin:
  1.	Se conecta a Redis y carga la memoria.
  2.	Registra su handler en nucleus.comando_handlers["comm_system"].
  3.	Inicia una tarea asíncrona que lee de stream_in y publica en stream_out.

Enviar comandos

Envía JSON al stream de entrada (stream_in), por ejemplo desde un cliente Python:

import asyncio, json
from aioredis import from_url

async def enviar_comando(cmd):
    r = await from_url("redis://user:pass@localhost:6379", decode_responses=True)
    await r.xadd("corec_commands", {"data": json.dumps(cmd)})
    # Leer respuesta
    resp = await r.xread({"corec_responses": "0-0"}, count=1, block=5000)
    print(resp)

asyncio.run(enviar_comando({
    "action": "chat",
    "params": {"mensaje": "¿Cuál es el estado de los bloques?"}
}))

Ejemplos de uso
  1.	Chat general

{"action":"chat","params":{"mensaje":"¡Hola! ¿Cómo va todo?"}}


  2.	Estado de CoreC

{"action":"status"}


  3.	Crear un plugin

{"action":"create_plugin","params":{"plugin_name":"sensor_monitor"}}



⸻

7. API de Redis Streams
  •	Entrada:
  •	Stream: comm_system.stream_in (por defecto corec_commands)
  •	Campo: data → JSON con { action, params }
  •	Salida:
  •	Stream: comm_system.stream_out (por defecto corec_responses)
  •	Campo: data → JSON con la respuesta del sistema

⸻

8. Extensiones futuras
  •	Generator completo para scaffolding de plugins.
  •	Controller de mantenimiento (backups, limpieza de logs, vacuum).
  •	Validación de esquemas con pydantic.
  •	Métricas y health‑checks HTTP.
  •	Soporte CLI con autocompletado y formato enriquecido.

⸻

9. Autores

Moises Alvarenga & Luna
CommSystem Plugin © 2025 — Todos los derechos reservados

⸻

