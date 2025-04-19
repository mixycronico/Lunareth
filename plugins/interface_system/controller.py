#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/interface_system/controller.py
Ejecuta acciones en CoreC para el plugin InterfaceSystem.
"""
import json
import time
import logging
from typing import Dict, Any
from corec.core import serializar_mensaje, deserializar_mensaje

class InterfaceController:
    def __init__(self, nucleus, redis_client):
        self.nucleus = nucleus
        self.redis_client = redis_client
        self.logger = logging.getLogger("InterfaceController")
        self.input_stream = "user_input_stream"
        self.output_stream = "user_output_stream"

    async def activar_plugin(self, nombre: str) -> str:
        try:
            config_path = f"plugins/{nombre}/config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            config["enabled"] = True
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            self.nucleus.cargar_plugins()
            return f"Plugin '{nombre}' activado correctamente. 🚀"
        except Exception as e:
            self.logger.error(f"Error activando plugin {nombre}: {e}")
            return f"Error activando plugin '{nombre}': {str(e)}"

    async def desactivar_plugin(self, nombre: str) -> str:
        try:
            config_path = f"plugins/{nombre}/config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            config["enabled"] = False
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            if nombre in self.nucleus.plugins:
                await self.nucleus.plugins[nombre].detener()
                del self.nucleus.plugins[nombre]
            return f"Plugin '{nombre}' desactivado correctamente. 🛑"
        except Exception as e:
            self.logger.error(f"Error desactivando plugin {nombre}: {e}")
            return f"Error desactivando plugin '{nombre}': {str(e)}"

    async def reiniciar_corec(self) -> str:
        try:
            await self.nucleus.detener()
            await self.nucleus.inicializar()
            return "CoreC reiniciado correctamente. 🔄"
        except Exception as e:
            self.logger.error(f"Error reiniciando CoreC: {e}")
            return f"Error reiniciando CoreC: {str(e)}"

    async def estado_sistema(self) -> Dict[str, Any]:
        try:
            bloques = self.nucleus.modulos["registro"].bloques
            estado = {
                "modulos_activos": list(self.nucleus.modulos.keys()),
                "plugins_activos": list(self.nucleus.plugins.keys()),
                "bloques": [
                    {"id": bid, "canal": b.canal, "fitness": b.fitness, "entidades": len(b.entidades)}
                    for bid, b in bloques.items()
                ],
                "alertas": self.nucleus.modulos["auditoria"].mem.get("alertas", []) if "auditoria" in self.nucleus.modulos else [],
                "nodos": self.nucleus.config.get("nodos", 1)
            }
            return estado
        except Exception as e:
            self.logger.error(f"Error consultando estado: {e}")
            return {"error": str(e)}

    async def listar_nodos(self) -> Dict[str, Any]:
        try:
            nodos = self.nucleus.config.get("nodos", 1)
            return {"nodos": [{"id": f"nodo_{i}", "activo": True} for i in range(nodos)]}
        except Exception as e:
            self.logger.error(f"Error listando nodos: {e}")
            return {"error": str(e)}

    async def listar_alertas(self) -> Dict[str, Any]:
        try:
            alertas = self.nucleus.modulos["auditoria"].mem.get("alertas", []) if "auditoria" in self.nucleus.modulos else []
            return {"alertas": alertas}
        except Exception as e:
            self.logger.error(f"Error listando alertas: {e}")
            return {"error": str(e)}

    async def configurar(self, clave: str, valor: Any) -> str:
        try:
            self.nucleus.config[clave] = valor
            self.nucleus._cargar_config(self.nucleus.config_path)
            return f"Configuración actualizada: {clave} = {valor} 🛠️"
        except Exception as e:
            self.logger.error(f"Error configurando: {e}")
            return f"Error configurando: {str(e)}"

    async def enviar_chat(self, mensaje: str, valor: float = 0.5) -> str:
        try:
            mensaje_data = {"texto": mensaje, "valor": valor}
            await self.redis_client.xadd(self.input_stream, {"data": json.dumps(mensaje_data)})
            start = time.time()
            while time.time() - start < 5:
                mensajes = await self.redis_client.xread({self.output_stream: "0-0"}, count=1)
                for _, entries in mensajes:
                    for _, data in entries:
                        if data.get("texto"):
                            return data["texto"]
                await asyncio.sleep(0.1)
            return "Timeout esperando respuesta ⏰"
        except Exception as e:
            self.logger.error(f"Error enviando chat: {e}")
            return f"Error en comunicación: {str(e)}"