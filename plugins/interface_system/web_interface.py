#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/interface_system/web_interface.py
Interfaz Web para el plugin InterfaceSystem.
"""
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
from brain import JarvisBrain
from controller import InterfaceController
import logging

app = FastAPI(title="CoreC Interface Web")
app.mount("/static", StaticFiles(directory="plugins/interface_system/static"), name="static")

class WebInterface:
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.logger = logging.getLogger("WebInterface")
        self.config = config
        self.brain = JarvisBrain(nucleus, use_redis=config.get("use_redis", True))
        self.redis_client = None
        self.controller = None

    async def inicializar(self):
        redis_url = f"redis://{self.nucleus.redis_config['username']}:{self.nucleus.redis_config['password']}@{self.nucleus.redis_config['host']}:{self.nucleus.redis_config['port']}"
        self.redis_client = await self.nucleus.aioredis.from_url(redis_url, decode_responses=True)
        self.controller = InterfaceController(self.nucleus, self.redis_client)
        await self.brain.inicializar_redis()
        self.logger.info("WebInterface inicializado")

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
                if comando == "activar plugin":
                    await ws.send_text(json.dumps({"message": "¿Qué plugin deseas activar? 🚀"}))
                elif comando.startswith("plugin "):
                    plugin = comando.split(" ")[1]
                    resp = await self.controller.activar_plugin(plugin)
                    await ws.send_text(json.dumps({"message": resp}))
                elif comando == "estado":
                    estado = await self.controller.estado_sistema()
                    await ws.send_text(json.dumps({"message": "Estado", "data": estado}))
                elif comando == "nodos":
                    nodos = await self.controller.listar_nodos()
                    await ws.send_text(json.dumps({"message": "Nodos", "data": nodos}))
                elif comando == "alertas":
                    alertas = await self.controller.listar_alertas()
                    await ws.send_text(json.dumps({"message": "Alertas", "data": alertas}))
                else:
                    respuesta = await self.controller.enviar_chat(comando)
                    self.brain.recordar(comando, respuesta)
                    await ws.send_text(json.dumps({"message": respuesta}))
            except Exception as e:
                self.logger.error(f"Error en WebSocket: {e}")
                await ws.send_text(json.dumps({"message": f"Error: {str(e)}"}))
                break

def inicializar(nucleus, config):
    web = WebInterface(nucleus, config)
    asyncio.create_task(web.inicializar())
    return app