# corec/nucleus.py
import logging
import importlib
import asyncio
from pathlib import Path
from typing import Dict, Any

from corec.core import ModuloBase, cargar_config
from corec.db import init_postgresql
from corec.redis_client import init_redis

class CoreCNucleus:
    def __init__(self, config_path: str):
        self.logger       = logging.getLogger("CoreCNucleus")
        self.config       = cargar_config(config_path)
        self.db_config    = self.config.get("db_config", {})
        self.redis_conf   = self.config.get("redis_config", {})
        self.redis_client = None

        self.modules      = {}             # Módulos internos
        self.plugins: Dict[str, Any] = {}  # Plugins registrados
        self.instance_id  = self.config.get("instance_id", "corec1")

    async def inicializar(self):
        # Inicializa PostgreSQL y Redis
        init_postgresql(self.db_config)
        self.redis_client = await init_redis(self.redis_conf)

        # Carga automática de módulos corec/modules/*.py
        mods_dir = Path(__file__).parent / "modules"
        for file in mods_dir.glob("*.py"):
            if file.name.startswith("__"):
                continue
            name = file.stem
            m    = importlib.import_module(f"corec.modules.{name}")
            cls  = getattr(m, f"Modulo{name.capitalize()}", None)
            if cls and issubclass(cls, ModuloBase):
                inst = cls()
                await inst.inicializar(self)
                self.modules[name] = inst
                self.logger.info(f"[Nucleus] módulo '{name}' listo")

    async def ejecutar(self):
        # Ejecuta todos los módulos corec en paralelo
        tareas = [m.ejecutar() for m in self.modules.values()]
        await asyncio.gather(*tareas, return_exceptions=True)

    async def detener(self):
        # Detiene todos los módulos corec
        for m in self.modules.values():
            await m.detener()
        # Cierra Redis
        if self.redis_client:
            await self.redis_client.close()
        self.logger.info("[Nucleus] detenido")

    async def publicar_alerta(self, alerta: dict):
        # Emite una alerta en logs (puedes extenderlo a Redis o a un stream)
        self.logger.warning(f"[Alerta] {alerta}")

    # ——— Nuevo soporte para Plugins ———

    def registrar_plugin(self, nombre: str, plugin: Any) -> None:
        """
        Registra una instancia de plugin bajo el nombre dado.
        """
        self.plugins[nombre] = plugin
        self.logger.info(f"[Nucleus] plugin '{nombre}' registrado")

    async def ejecutar_plugin(self, nombre: str, comando: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoca el método `manejar_comando` del plugin registrado.
        Devuelve el dict de respuesta que implemente el plugin.
        """
        plugin = self.plugins.get(nombre)
        if not plugin:
            raise ValueError(f"Plugin '{nombre}' no encontrado")
        if not hasattr(plugin, "manejar_comando"):
            raise AttributeError(f"Plugin '{nombre}' no implementa 'manejar_comando'")
        return await plugin.manejar_comando(comando)