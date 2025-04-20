import logging
import importlib
import asyncio
import json
import time
import random
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field, ValidationError

from corec.core import ModuloBase, cargar_config
from corec.db import init_postgresql
from corec.redis_client import init_redis
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad

class PluginBlockConfig(BaseModel):
    """Configuración de un bloque simbiótico para un plugin."""
    bloque_id: str = Field(..., regex=r"^[a-zA-Z0-9_-]+$")
    canal: int = Field(..., ge=1, le=10)
    entidades: int = Field(..., ge=100, le=10000)
    max_size_mb: int = Field(..., ge=1, le=10)
    max_errores: float = Field(..., ge=0.01, le=0.5)
    min_fitness: float = Field(..., ge=0.1, le=0.9)

class CoreCNucleus:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger("CoreCNucleus")
        self.config = cargar_config(config_path)
        self.db_config = self.config.get("db_config", {})
        self.redis_conf = self.config.get("redis_config", {})
        self.redis_client = None
        self.modules = {}  # Módulos internos
        self.plugins: Dict[str, Any] = {}  # Plugins registrados
        self.bloques_plugins: Dict[str, BloqueSimbiotico] = {}  # Bloques por plugin
        self.instance_id = self.config.get("instance_id", "corec1")

    async def inicializar(self):
        """Inicializa PostgreSQL, Redis, módulos y bloques de plugins."""
        init_postgresql(self.db_config)
        self.redis_client = await init_redis(self.redis_conf)
        # Carga módulos
        mods_dir = Path(__file__).parent / "modules"
        for file in mods_dir.glob("*.py"):
            if file.name.startswith("__"):
                continue
            name = file.stem
            m = importlib.import_module(f"corec.modules.{name}")
            cls = getattr(m, f"Modulo{name.capitalize()}", None)
            if cls and issubclass(cls, ModuloBase):
                inst = cls()
                await inst.inicializar(self)
                self.modules[name] = inst
                self.logger.info(f"[Nucleus] módulo '{name}' listo")

    async def ejecutar(self):
        """Ejecuta módulos y coordina bloques de plugins."""
        tareas = [m.ejecutar() for m in self.modules.values()]
        tareas.append(self.coordinar_bloques())
        await asyncio.gather(*tareas, return_exceptions=True)

    async def detener(self):
        """Detiene módulos, bloques y cierra Redis."""
        for m in self.modules.values():
            await m.detener()
        for bloque in self.bloques_plugins.values():
            await bloque.escribir_postgresql(self.db_config)
        if self.redis_client:
            await self.redis_client.close()
        self.logger.info("[Nucleus] detenido")

    async def publicar_alerta(self, alerta: dict):
        """Publica una alerta en logs y en el stream corec_alerts."""
        self.logger.warning(f"[Alerta] {alerta}")
        try:
            await self.redis_client.xadd("corec_alerts", {"data": json.dumps(alerta)})
        except Exception as e:
            self.logger.error(f"[Alerta] Error publicando en corec_alerts: {e}")

    def registrar_plugin(self, nombre: str, plugin: Any) -> None:
        """Registra un plugin y asigna un bloque simbiótico."""
        self.plugins[nombre] = plugin
        # Carga configuración del bloque desde plugins[<nombre>]["bloque"]
        plugin_conf = self.config.get("plugins", {}).get(nombre, {})
        bloque_conf = plugin_conf.get("bloque")
        if bloque_conf:
            try:
                cfg = PluginBlockConfig(**bloque_conf)
                entidades = []
                for i in range(cfg.entidades):
                    async def tmp(): return {"valor": random.random()}
                    entidades.append(crear_entidad(f"m{time.time_ns()}_{i}", cfg.canal, tmp))
                bloque = BloqueSimbiotico(
                    id=cfg.bloque_id,
                    canal=cfg.canal,
                    entidades=entidades,
                    max_size_mb=cfg.max_size_mb,
                    nucleus=self
                )
                self.bloques_plugins[nombre] = bloque
                self.logger.info(f"[Nucleus] Bloque '{cfg.bloque_id}' asignado al plugin '{nombre}'")
            except ValidationError as e:
                self.logger.error(f"[Nucleus] Configuración de bloque inválida para '{nombre}': {e}")
            except Exception as e:
                self.logger.error(f"[Nucleus] Error asignando bloque para '{nombre}': {e}")
        self.logger.info(f"[Nucleus] plugin '{nombre}' registrado")

    async def ejecutar_plugin(self, nombre: str, comando: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta un comando en el plugin, validando con pydantic."""
        plugin = self.plugins.get(nombre)
        if not plugin:
            raise ValueError(f"Plugin '{nombre}' no encontrado")
        if not hasattr(plugin, "manejar_comando"):
            raise AttributeError(f"Plugin '{nombre}' no implementa 'manejar_comando'")
        # Valida comando básico
        class PluginCommand(BaseModel):
            action: str
            params: Dict[str, Any] = {}
        try:
            cmd = PluginCommand(**comando)
            return await plugin.manejar_comando(cmd.dict())
        except ValidationError as e:
            self.logger.error(f"[Nucleus] Comando inválido para '{nombre}': {e}")
            return {"status": "error", "message": f"Comando inválido: {e}"}

    async def coordinar_bloques(self):
        """Coordina bloques simbióticos, priorizando tareas activas."""
        while True:
            try:
                for nombre, bloque in self.bloques_plugins.items():
                    carga = random.random()  # TODO: Calcular carga real
                    await bloque.procesar(carga)
                    self.logger.debug(f"[Nucleus] Bloque '{bloque.id}' procesado, fitness: {bloque.fitness:.2f}")
            except Exception as e:
                self.logger.error(f"[Nucleus] Error coordinando bloques: {e}")
                await self.publicar_alerta({"tipo": "error_coordinacion", "mensaje": str(e)})
            await asyncio.sleep(60)  # Coordinar cada 60 segundos
