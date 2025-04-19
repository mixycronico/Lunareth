#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/interface_system/brain.py
Núcleo conversacional para el plugin InterfaceSystem, integrado con ComunicadorInteligente.
"""
import json
import os
import time
from typing import Dict, Any
import logging
from corec.core import zstd

MEM_FILE = "plugins/interface_system/memory.json"
MEM_KEY = "interface_memory"

class JarvisBrain:
    def __init__(self, nucleus=None, use_redis: bool = True):
        self.nucleus = nucleus
        self.logger = logging.getLogger("JarvisBrain")
        self.use_redis = use_redis
        self.redis_client = None
        self.mem = self._cargar_memoria()

    async def inicializar_redis(self):
        if self.use_redis and self.nucleus:
            redis_url = f"redis://{self.nucleus.redis_config['username']}:{self.nucleus.redis_config['password']}@{self.nucleus.redis_config['host']}:{self.nucleus.redis_config['port']}"
            self.redis_client = await self.nucleus.aioredis.from_url(redis_url, decode_responses=False)
            self.logger.info("Redis inicializado para memoria")

    def _cargar_memoria(self) -> Dict[str, Any]:
        if self.use_redis and self.redis_client:
            try:
                cached = self.redis_client.get(MEM_KEY)
                if cached:
                    return json.loads(zstd.decompress(cached))
            except Exception as e:
                self.logger.error(f"Error cargando memoria desde Redis: {e}")
        if os.path.exists(MEM_FILE):
            try:
                with open(MEM_FILE, "r") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error cargando memoria desde archivo: {e}")
        return {"conversacion": [], "estado": {}, "config": {}, "timestamp": time.time()}

    def guardar_memoria(self):
        try:
            now = time.time()
            self.mem["conversacion"] = [
                c for c in self.mem["conversacion"]
                if c.get("timestamp", now) > now - 3600
            ]
            self.mem["timestamp"] = now
            if self.use_redis and self.redis_client:
                compressed = zstd.compress(json.dumps(self.mem).encode())
                self.redis_client.set(MEM_KEY, compressed, ex=86400)
            with open(MEM_FILE, "w") as f:
                json.dump(self.mem, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error guardando memoria: {e}")

    def recordar(self, pregunta: str, respuesta: str):
        self.mem["conversacion"].append({
            "q": pregunta,
            "r": respuesta,
            "timestamp": time.time()
        })
        if len(self.mem["conversacion"]) > 20:
            self.mem["conversacion"].pop(0)
        self.guardar_memoria()

    def actualizar_estado(self, clave: str, valor: Any):
        self.mem["estado"][clave] = valor
        self.guardar_memoria()

    def decidir(self, entrada: str) -> str:
        entrada = entrada.lower()
        contexto = [c["q"] for c in self.mem["conversacion"][-3:]]
        if "activar plugin" in entrada:
            return "¿Qué plugin deseas activar? 🚀"
        if "desactivar plugin" in entrada:
            return "¿Qué plugin deseas desactivar? 🛑"
        if "estado" in entrada or "status" in entrada:
            return "Consultando estado del sistema... 📊"
        if "reiniciar" in entrada or "restart" in entrada:
            return "Reiniciando CoreC... 🔄"
        if "nodos" in entrada or "nodes" in entrada:
            return "Listando nodos activos... 🌐"
        if "chat" in entrada:
            return "Enviando mensaje al sistema... 💬"
        if "config" in entrada:
            return "Actualizando configuración... 🛠️"
        if "alertas" in entrada or "alerts" in entrada:
            return "Mostrando alertas recientes... ⚠️"
        if any("activar plugin" in c for c in contexto) and entrada.strip():
            return f"Intentando activar plugin '{entrada}'..."
        if any("desactivar plugin" in c for c in contexto) and entrada.strip():
            return f"Intentando desactivar plugin '{entrada}'..."
        return "¡Estoy listo, bb! 😎 ¿Qué necesitas que haga con CoreC?"