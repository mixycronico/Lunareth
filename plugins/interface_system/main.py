#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/interface_system/main.py
Plugin que proporciona una CLI y WebSocket para CoreC, con vida via ComunicadorInteligente.
"""
import asyncio
import logging
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
import click
from corec.core import aioredis
from corec.entities import crear_entidad
from corec.blocks import BloqueSimbiotico
from brain import JarvisBrain
from controller import InterfaceController
from typing import Dict, Any

console = Console()

class InterfaceSystem:
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.logger = logging.getLogger("InterfaceSystem")
        self.config = config
        self.canal = config.get("canal", 8)
        self.bloque = None
        self.brain = JarvisBrain(nucleus, use_redis=config.get("use_redis", True))
        self.redis_client = None
        self.controller = None

    async def inicializar(self):
        # Inicializar Redis
        redis_url = f"redis://{self.nucleus.redis_config['username']}:{self.nucleus.redis_config['password']}@{self.nucleus.redis_config['host']}:{self.nucleus.redis_config['port']}"
        self.redis_client = await aioredis.from_url(redis_url, decode_responses=True)
        self.logger.info("Redis inicializado para InterfaceSystem")
        await self.brain.inicializar_redis()

        # Inicializar controlador
        self.controller = InterfaceController(self.nucleus, self.redis_client)

        # Crear bloque simbiótico
        entidades = [crear_entidad(f"m{i}", self.canal, self._procesar_comando) for i in range(self.config.get("entidades", 100))]
        self.bloque = BloqueSimbiotico(f"interface_system", self.canal, entidades, max_size=1024, nucleus=self.nucleus)
        self.nucleus.modulos["registro"].bloques[self.bloque.id] = self.bloque
        self.nucleus.registrar_plugin("interface_system", self)
        self.logger.info(f"Plugin InterfaceSystem inicializado con {len(entidades)} entidades")

        # Iniciar CLI
        asyncio.create_task(self._iniciar_cli())

    async def _procesar_comando(self, mensaje: Dict[str, Any] = None) -> Dict[str, Any]:
        if mensaje is None:
            mensaje = {"comando": "status"}
        comando = mensaje.get("comando", "").lower()
        valor = mensaje.get("valor", 0.5)
        texto = mensaje.get("texto", "")

        if comando == "status" or comando == "estado":
            estado = await self.controller.estado_sistema()
            if "error" in estado:
                return {"valor": 0.0, "texto": estado["error"]}
            table = Table(title="Estado del Sistema", style="cyan")
            table.add_column("Categoría", style="green")
            table.add_column("Detalles", style="white")
            table.add_row("Módulos Activos", ", ".join(estado["modulos_activos"]))
            table.add_row("Plugins Activos", ", ".join(estado["plugins_activos"]))
            table.add_row("Bloques", f"{len(estado['bloques'])} activos")
            table.add_row("Alertas", f"{len(estado['alertas'])} pendientes")
            table.add_row("Nodos", str(estado["nodos"]))
            console.print(table)
            respuesta = "Estado mostrado correctamente 🌱"
        elif comando == "plugins":
            estado = await self.controller.estado_sistema()
            if "error" in estado:
                return {"valor": 0.0, "texto": estado["error"]}
            table = Table(title="Plugins Activos", style="cyan")
            table.add_column("Nombre", style="green")
            for plugin in estado["plugins_activos"]:
                table.add_row(plugin)
            console.print(table)
            respuesta = f"{len(estado['plugins_activos'])} plugins activos 🎉"
        elif comando == "blocks" or comando == "bloques":
            estado = await self.controller.estado_sistema()
            if "error" in estado:
                return {"valor": 0.0, "texto": estado["error"]}
            table = Table(title="Bloques Simbióticos", style="cyan")
            table.add_column("ID", style="green")
            table.add_column("Canal", style="white")
            table.add_column("Fitness", style="white")
            table.add_column("Entidades", style="white")
            for bloque in estado["bloques"]:
                table.add_row(bloque["id"], str(bloque["canal"]), f"{bloque['fitness']:.2f}", str(bloque["entidades"]))
            console.print(table)
            respuesta = f"{len(estado['bloques'])} bloques activos 🧬"
        elif comando == "nodes" or comando == "nodos":
            nodos = await self.controller.listar_nodos()
            if "error" in nodos:
                return {"valor": 0.0, "texto": nodos["error"]}
            table = Table(title="Nodos Activos", style="cyan")
            table.add_column("ID", style="green")
            table.add_column("Estado", style="white")
            for nodo in nodos["nodos"]:
                table.add_row(nodo["id"], "Activo" if nodo["activo"] else "Inactivo")
            console.print(table)
            respuesta = f"{len(nodos['nodos'])} nodos activos 🌐"
        elif comando == "alerts" or comando == "alertas":
            alertas = await self.controller.listar_alertas()
            if "error" in alertas:
                return {"valor": 0.0, "texto": alertas["error"]}
            table = Table(title="Alertas Recientes", style="cyan")
            table.add_column("Alerta", style="green")
            for alerta in alertas["alertas"]:
                table.add_row(alerta)
            console.print(table)
            respuesta = f"{len(alertas['alertas'])} alertas pendientes ⚠️"
        elif comando.startswith("chat "):
            respuesta = await self.controller.enviar_chat(comando[5:], valor)
        elif comando.startswith("activar plugin "):
            plugin = comando.split("activar plugin ")[1].strip()
            respuesta = await self.controller.activar_plugin(plugin)
        elif comando.startswith("desactivar plugin "):
            plugin = comando.split("desactivar plugin ")[1].strip()
            respuesta = await self.controller.deactivar_plugin(plugin)
        elif comando.startswith("config "):
            partes = comando.split(" ", 2)
            if len(partes) < 3:
                return {"valor": 0.0, "texto": "Uso: config <clave> <valor>"}
            clave, valor = partes[1], partes[2]
            try:
                valor = json.loads(valor)
            except:
                pass
            respuesta = await self.controller.configurar(clave, valor)
        else:
            # Enviar a ComunicadorInteligente para respuestas vivas
            respuesta = await self.controller.enviar_chat(comando, valor)
        console.print(f"[bold magenta]CoreC[/bold magenta]: {respuesta}", style="green")
        self.brain.recordar(comando, respuesta)
        return {"valor": valor, "texto": respuesta}

    async def _iniciar_cli(self):
        @click.group()
        def cli():
            """CoreC Interface CLI - Asistente vivo para CoreC"""
            console.print(Panel(
                "[bold green]Bienvenido a CoreC Interface CLI[/bold green]\n"
                "Sistema bioinspirado ligero y escalable 🌱",
                border_style="cyan", expand=False
            ))

        @cli.command()
        def status():
            """Muestra el estado del sistema"""
            resultado = asyncio.run(self._procesar_comando({"comando": "status"}))
            self.brain.recordar("status", resultado["texto"])

        @cli.command()
        def plugins():
            """Lista los plugins activos"""
            resultado = asyncio.run(self._procesar_comando({"comando": "plugins"}))
            self.brain.recordar("plugins", resultado["texto"])

        @cli.command()
        def blocks():
            """Lista los bloques simbióticos"""
            resultado = asyncio.run(self._procesar_comando({"comando": "blocks"}))
            self.brain.recordar("blocks", resultado["texto"])

        @cli.command()
        def nodes():
            """Lista los nodos activos"""
            resultado = asyncio.run(self._procesar_comando({"comando": "nodes"}))
            self.brain.recordar("nodes", resultado["texto"])

        @cli.command()
        def alerts():
            """Muestra las alertas recientes"""
            resultado = asyncio.run(self._procesar_comando({"comando": "alerts"}))
            self.brain.recordar("alerts", resultado["texto"])

        @cli.command()
        @click.argument("mensaje")
        def chat(mensaje):
            """Envía un mensaje al sistema"""
            resultado = asyncio.run(self._procesar_comando({"comando": f"chat {mensaje}"}))
            self.brain.recordar(f"chat {mensaje}", resultado["texto"])

        @cli.command()
        @click.argument("plugin")
        def activate(plugin):
            """Activa un plugin"""
            resultado = asyncio.run(self._procesar_comando({"comando": f"activar plugin {plugin}"}))
            self.brain.recordar(f"activar plugin {plugin}", resultado["texto"])

        @cli.command()
        @click.argument("plugin")
        def deactivate(plugin):
            """Desactiva un plugin"""
            resultado = asyncio.run(self._procesar_comando({"comando": f"desactivar plugin {plugin}"}))
            self.brain.recordar(f"desactivar plugin {plugin}", resultado["texto"])

        @cli.command()
        @click.argument("clave")
        @click.argument("valor")
        def config(clave, valor):
            """Actualiza una configuración"""
            resultado = asyncio.run(self._procesar_comando({"comando": f"config {clave} {valor}"}))
            self.brain.recordar(f"config {clave} {valor}", resultado["texto"])

        cli()

    async def ejecutar(self):
        while True:
            resultado = await self.bloque.procesar(self.config.get("carga", 0.5))
            await self.nucleus.publicar_alerta({
                "tipo": "interface_actividad",
                "bloque_id": self.bloque.id,
                "fitness": resultado["fitness"],
                "timestamp": time.time()
            })
            await asyncio.sleep(self.config.get("intervalo", 60))

    async def detener(self):
        self.logger.info("Plugin InterfaceSystem detenido")
        if self.redis_client:
            await self.redis_client.close()
        self.brain.guardar_memoria()

def inicializar(nucleus, config):
    plugin = InterfaceSystem(nucleus, config)
    asyncio.create_task(plugin.inicializar())