from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import asyncio
import logging

from brain import JarvisBrain
from controller import InterfaceController


app = FastAPI(title="CoreC Interface Web")
app.mount("/static", StaticFiles(directory="plugins/interface_system/static"), name="static")

# Contexto global para acceder al controlador y al cerebro
contexto_web_interface = {
    "controller": None,
    "brain": None,
    "logger": logging.getLogger("WebInterface")
}


class WebInterface:
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.config = config
        self.brain = JarvisBrain(nucleus, use_redis=config.get("use_redis", True))
        self.redis_client = None
        self.controller = None

    async def inicializar(self):
        redis_cfg = self.nucleus.redis_config
        redis_url = (
            f"redis://{redis_cfg['username']}:{redis_cfg['password']}"
            f"@{redis_cfg['host']}:{redis_cfg['port']}"
        )
        self.redis_client = await self.nucleus.aioredis.from_url(
            redis_url, decode_responses=True
        )
        self.controller = InterfaceController(self.nucleus, self.redis_client)
        await self.brain.inicializar_redis()

        contexto_web_interface["controller"] = self.controller
        contexto_web_interface["brain"] = self.brain
        contexto_web_interface["logger"] = logging.getLogger("WebInterface")
        contexto_web_interface["logger"].info("WebInterface inicializado")


# ---- Rutas del servidor WebSocket y HTML ---- #

@app.get("/")
async def get_index():
    with open("plugins/interface_system/static/index.html") as f:
        return HTMLResponse(content=f.read())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_text(json.dumps({"message": "Hola, soy CoreC 🌱"}))

    while True:
        try:
            data = await ws.receive_text()
            comando = json.loads(data).get("comando", "").lower()
            controller = contexto_web_interface["controller"]
            brain = contexto_web_interface["brain"]

            if comando == "activar plugin":
                await ws.send_text(json.dumps({
                    "message": "¿Qué plugin deseas activar? 🚀"
                }))
            elif comando.startswith("plugin "):
                plugin = comando.split(" ")[1]
                resp = await controller.activar_plugin(plugin)
                await ws.send_text(json.dumps({"message": resp}))
            elif comando == "estado":
                estado = await controller.estado_sistema()
                await ws.send_text(json.dumps({
                    "message": "Estado", "data": estado
                }))
            elif comando == "nodos":
                nodos = await controller.listar_nodos()
                await ws.send_text(json.dumps({
                    "message": "Nodos", "data": nodos
                }))
            elif comando == "alertas":
                alertas = await controller.listar_alertas()
                await ws.send_text(json.dumps({
                    "message": "Alertas", "data": alertas
                }))
            else:
                respuesta = await controller.enviar_chat(comando)
                brain.recordar(comando, respuesta)
                await ws.send_text(json.dumps({"message": respuesta}))

        except Exception as e:
            contexto_web_interface["logger"].error(f"Error en WebSocket: {e}")
            await ws.send_text(json.dumps({"message": f"Error: {str(e)}"}))
            break


# ---- Función para inicializar ---- #

def inicializar(nucleus, config):
    web = WebInterface(nucleus, config)
    asyncio.create_task(web.inicializar())
    return app
