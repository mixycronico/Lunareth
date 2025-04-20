# 🌱 CommSystem

**Tu asistente conversacional vivo para CoreC**  
Chat real, memoria persistente y generación de plugins on‑demand.

---

## ✨ Características

- **Chat dinámico** con `google/gemini-2.0-flash-lite-001` vía OpenRouter  
- **Memoria inteligente** (hasta 50 intercambios) almacenada en Redis y respaldo en disco  
- **Comandos por Redis Streams** (sincronía total, sin HTTP/WebSocket)  
- **Scaffolding on‑demand**: crea nuevos plugins con un solo comando  
- **Modo mantenimiento**: consulta estado, lista módulos/plugins y más  

---

## 🚀 Instalación

1. **Copia** el plugin en tu árbol de CoreC:
   ```bash
   cp -R comm_system plugins/comm_system

  2.	Instala dependencias:

pip install -r plugins/comm_system/requirements.txt


  3.	Configura tu API‑Key de OpenRouter:

export OPENROUTER_API_KEY="tu_api_key"


  4.	Reinicia CoreC:

bash run.sh



⸻

⚙️ Configuración

Edita plugins/comm_system/config.json:

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
  •	memory_ttl: duración de la memoria en segundos.
  •	memory_file: ruta de respaldo JSON.

⸻

📚 Uso

Envía JSON al stream de entrada y recibe la respuesta por el stream de salida.

Ejemplo en Python

import asyncio, json
from aioredis import from_url

async def enviar_comando(cmd):
    r = await from_url("redis://:password@localhost:6379", decode_responses=True)
    await r.xadd("corec_commands", {"data": json.dumps(cmd)})
    # Espera y lee la respuesta
    resp = await r.xread({"corec_responses": "0-0"}, count=1, block=5000)
    print("Respuesta:", json.loads(resp[0][1][0][1]["data"]))

# Chat
asyncio.run(enviar_comando({
    "action": "chat",
    "params": {"mensaje": "¿Cómo va el sistema?"}
}))

# Estado de CoreC
asyncio.run(enviar_comando({ "action": "status" }))



⸻

🛠️ Desarrollo
  •	Código fuente en plugins/comm_system/
  •	Processors:
  •	memory.py → gestión de historial
  •	ai_chat.py → llamadas a OpenRouter
  •	manager.py → loop de comandos
  •	Utils: helpers.py para I/O asíncrono
  •	Tests: crea tus casos en tests/test_comm_system.py

⸻

🤝 Contribuir
  1.	Haz un fork del repositorio.
  2.	Crea tu rama de feature: git checkout -b feature/nombre
  3.	Asegúrate de pasar lint y tests:

black . && flake8 && pytest -q


  4.	Envía un pull request describiendo tus cambios.

⸻

📄 Licencia

MIT © 2025 Moises Alvarenga & Luna

⸻
