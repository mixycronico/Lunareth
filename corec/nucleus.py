import logging
import yaml
import aioredis
import asyncpg
from pydantic import ValidationError
from typing import Dict, Any
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from corec.entities import crear_entidad
from corec.blocks import BloqueSimbiotico
from plugins import PluginBlockConfig, PluginCommand


def cargar_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


async def init_postgresql(config: Dict[str, Any]):
    return await asyncpg.connect(**config)


class CoreCNucleus:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger("CoreCNucleus")
        self.config_path = config_path
        self.config = None
        self.db_pool = None
        self.redis_client = None
        self.modules = {}
        self.plugins = {}

    async def inicializar(self):
        try:
            self.config = cargar_config(self.config_path)
            self.db_pool = await init_postgresql(self.config["db_config"])
            self.redis_client = await aioredis.from_url(
                f"redis://{self.config['redis_config']['host']}:{self.config['redis_config']['port']}"
            )
            self.modules["registro"] = ModuloRegistro()
            self.modules["sincronizacion"] = ModuloSincronizacion()
            self.modules["ejecucion"] = ModuloEjecucion()
            self.modules["auditoria"] = ModuloAuditoria()
            for name, module in self.modules.items():
                await module.inicializar(self, self.config.get(f"{name}_config"))
            for block_config in self.config.get("bloques", []):
                try:
                    block_config = PluginBlockConfig(**block_config)
                    entidades = [crear_entidad(f"ent_{i}", block_config.canal, lambda carga: {"valor": 0.5})
                                 for i in range(block_config.entidades)]
                    bloque = BloqueSimbiotico(
                        block_config.id,
                        block_config.canal,
                        entidades,
                        block_config.max_size_mb if hasattr(block_config, "max_size_mb") else 10.0,
                        self
                    )
                    await self.modules["registro"].registrar_bloque(
                        block_config.id,
                        block_config.canal,
                        block_config.entidades,
                        block_config.max_size_mb if hasattr(block_config, "max_size_mb") else 10.0
                    )
                except ValidationError as e:
                    await self.publicar_alerta({
                        "tipo": "error_config_bloque",
                        "bloque_id": block_config.id,
                        "mensaje": str(e),
                        "timestamp": random.random()
                    })
            self.logger.info("[Núcleo] Inicialización completa")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error inicializando: {e}")
            if "redis" in str(e).lower():
                self.redis_client = None

    def registrar_plugin(self, plugin_id: str, plugin: Any):
        try:
            if self.config is None:
                self.logger.warning("[Núcleo] Configuración no inicializada, registrando plugin sin configuración")
                self.plugins[plugin_id] = plugin
                self.logger.info(f"[Núcleo] Plugin '{plugin_id}' registrado sin configuración")
                return
            plugin_config = self.config.get("plugins", {}).get(plugin_id)
            if plugin_config:
                block_config = PluginBlockConfig(**plugin_config["bloque"])
                entidades = [crear_entidad(f"ent_{i}", block_config.canal, lambda carga: {"valor": 0.5})
                             for i in range(block_config.entidades)]
                bloque = BloqueSimbiotico(
                    block_config.id,
                    block_config.canal,
                    entidades,
                    block_config.max_size_mb if hasattr(block_config, "max_size_mb") else 10.0,
                    self
                )
                plugin.bloque = bloque
            self.plugins[plugin_id] = plugin
            self.logger.info(f"[Núcleo] Plugin '{plugin_id}' registrado")
        except ValidationError as e:
            self.logger.error(f"[Núcleo] Configuración inválida para plugin '{plugin_id}': {e}")
            self.plugins[plugin_id] = plugin  # Registrar el plugin incluso si la configuración falla
            self.logger.info(f"[Núcleo] Plugin '{plugin_id}' registrado a pesar de configuración inválida")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error registrando plugin '{plugin_id}': {e}")
            self.plugins[plugin_id] = plugin  # Registrar el plugin incluso si hay un error

    async def ejecutar_plugin(self, plugin_id: str, comando: Dict[str, Any]) -> Dict[str, Any]:
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin '{plugin_id}' no encontrado")
        try:
            comando = PluginCommand(**comando)
            result = await plugin.manejar_comando(comando)
            self.logger.info(f"[Núcleo] Comando ejecutado en plugin '{plugin_id}'")
            return result
        except ValidationError as e:
            self.logger.error(f"[Núcleo] Comando inválido para plugin '{plugin_id}': {e}")
            return {"status": "error", "message": str(e)}
        except Exception as e:
            self.logger.error(f"[Núcleo] Error ejecutando comando en plugin '{plugin_id}': {e}")
            return {"status": "error", "message": str(e)}

    async def publicar_alerta(self, alerta: Dict[str, Any]):
        try:
            if not self.redis_client:
                self.logger.warning("[Alerta] Redis client no inicializado, alerta no publicada")
                return
            stream_key = f"alertas:{alerta['tipo']}"
            await self.redis_client.xadd(stream_key, alerta)
            self.logger.warning(f"[Alerta] {alerta}")
        except Exception as e:
            self.logger.error(f"[Alerta] Error publicando alerta: {e}")

    async def detener(self):
        for module in self.modules.values():
            await module.detener()
        if self.redis_client:
            await self.redis_client.close()
        if self.db_pool:
            await self.db_pool.close()
        self.logger.info("[Núcleo] Detención completa")
