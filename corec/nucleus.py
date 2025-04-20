import logging
import random
import aioredis
from typing import Dict
from corec.core import ComponenteBase, cargar_config, PluginBlockConfig, PluginCommand
from corec.db import init_postgresql
from corec.blocks import BloqueSimbiotico
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from pydantic import ValidationError
from corec.entities import crear_entidad


class CoreCNucleus:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger("CoreCNucleus")
        self.config = cargar_config(config_path)
        self.db_config = self.config.get("db_config", {})
        self.redis_config = self.config.get("redis_config", {})
        self.redis_client = None
        self.modules: Dict[str, ComponenteBase] = {}
        self.plugins: Dict[str, ComponenteBase] = {}
        self.bloques_plugins: Dict[str, BloqueSimbiotico] = {}

    async def inicializar(self):
        """Inicializa el núcleo, configurando base de datos, módulos y plugins."""
        try:
            self.logger.info("[Nucleus] Inicializando núcleo")
            init_postgresql(self.db_config)
            if self.redis_config:
                try:
                    self.redis_client = await aioredis.from_url(
                        f"redis://{self.redis_config['host']}:{self.redis_config['port']}",
                        username=self.redis_config.get("username"),
                        password=self.redis_config.get("password")
                    )
                except Exception as e:
                    self.logger.error(f"[Nucleus] Error conectando a Redis: {e}")
                    self.redis_client = None

            # Inicializar módulos
            self.modules = {
                "registro": ModuloRegistro(),
                "sincronizacion": ModuloSincronizacion(),
                "ejecucion": ModuloEjecucion(),
                "auditoria": ModuloAuditoria()
            }
            for nombre, modulo in self.modules.items():
                await modulo.inicializar(self)
                self.logger.info(f"[Nucleus] Módulo '{nombre}' inicializado")

            # Cargar bloques desde config
            for bloque_conf in self.config.get("bloques", []):
                try:
                    config = PluginBlockConfig(**bloque_conf)
                    entidades = [crear_entidad(f"ent_{i}", config.canal, lambda: {"valor": random.uniform(0, 1)}) for i in range(config.entidades)]
                    bloque = BloqueSimbiotico(config.id, config.canal, entidades, self)
                    await self.modules["registro"].registrar_bloque(
                        config.id, config.canal, config.entidades
                    )
                except ValidationError as e:
                    self.logger.error(f"[Nucleus] Configuración inválida para bloque: {e}")
                    await self.publicar_alerta({
                        "tipo": "error_registro",
                        "bloque_id": bloque_conf.get("id", "desconocido"),
                        "mensaje": str(e),
                        "timestamp": random.random()
                    })

            # Cargar plugins desde config
            for nombre, conf in self.config.get("plugins", {}).items():
                if conf.get("enabled", False):
                    try:
                        bloque_conf = conf.get("bloque", {})
                        config = PluginBlockConfig(**bloque_conf)
                        entidades = [crear_entidad(f"ent_{i}", config.canal, lambda: {"valor": random.uniform(0, 1)}) for i in range(config.entidades)]
                        bloque = BloqueSimbiotico(config.bloque_id, config.canal, entidades, self)
                        self.bloques_plugins[nombre] = bloque
                        await self.modules["registro"].registrar_bloque(
                            config.bloque_id, config.canal, config.entidades
                        )
                    except ValidationError as e:
                        self.logger.error(f"[Nucleus] Configuración de bloque inválida para '{nombre}': {e}")

        except Exception as e:
            self.logger.error(f"[Nucleus] Error inicializando núcleo: {e}")

    def registrar_plugin(self, nombre: str, plugin: ComponenteBase):
        """Registra un plugin en el núcleo."""
        try:
            self.plugins[nombre] = plugin
            conf = self.config.get("plugins", {}).get(nombre, {})
            bloque_conf = conf.get("bloque", {})
            if bloque_conf:
                try:
                    config = PluginBlockConfig(**bloque_conf)
                    entidades = [crear_entidad(f"ent_{i}", config.canal, lambda: {"valor": random.uniform(0, 1)}) for i in range(config.entidades)]
                    bloque = BloqueSimbiotico(config.bloque_id, config.canal, entidades, self)
                    self.bloques_plugins[nombre] = bloque
                    self.logger.info(f"[Nucleus] Plugin '{nombre}' registrado con bloque")
                except ValidationError as e:
                    self.logger.error(f"[Nucleus] Configuración de bloque inválida para '{nombre}': {e}")
            else:
                self.logger.info(f"[Nucleus] Plugin '{nombre}' registrado sin bloque")
        except Exception as e:
            self.logger.error(f"[Nucleus] Error registrando plugin '{nombre}': {e}")

    async def ejecutar_plugin(self, nombre: str, comando: dict) -> dict:
        """Ejecuta un comando en un plugin registrado."""
        if nombre not in self.plugins:
            raise ValueError(f"Plugin '{nombre}' no encontrado")
        try:
            cmd = PluginCommand(**comando)
            resultado = await self.plugins[nombre].manejar_comando(cmd)
            self.logger.info(f"[Nucleus] Comando ejecutado en '{nombre}': {comando}")
            return resultado  # Retornar resultado
        except ValidationError as e:
            self.logger.error(f"[Nucleus] Comando inválido para '{nombre}': {e}")
            return {"status": "error", "message": f"Comando inválido: {e}"}
        except Exception as e:
            self.logger.error(f"[Nucleus] Error ejecutando comando en '{nombre}': {e}")
            return {"status": "error", "message": str(e)}

    async def publicar_alerta(self, alerta: dict):
        """Publica una alerta en el canal corec_alerts."""
        try:
            self.logger.warning(f"[Alerta] {alerta}")
            if self.redis_client:
                await self.redis_client.xadd("corec_alerts", alerta)
            else:
                self.logger.error("[Alerta] Redis client no inicializado")
        except Exception as e:
            self.logger.error(f"[Alerta] Error publicando en corec_alerts: {e}")

    async def coordinar_bloques(self):
        """Coordina bloques simbióticos, priorizando tareas activas."""
        try:
            for nombre, bloque in self.bloques_plugins.items():
                carga = random.random()  # TODO: Calcular carga real
                await bloque.procesar(carga)
                self.logger.debug(
                    f"[Nucleus] Bloque '{bloque.id}' procesado, "
                    f"fitness: {bloque.fitness:.2f}"
                )
        except Exception as e:
            self.logger.error(f"[Nucleus] Error coordinando bloques: {e}")
            await self.publicar_alerta({
                "tipo": "error_coordinacion",
                "mensaje": str(e),
                "timestamp": random.random()
            })

    async def detener(self):
        """Detiene el núcleo y cierra conexiones."""
        try:
            for modulo in self.modules.values():
                await modulo.detener()
            if self.redis_client:
                await self.redis_client.close()
            self.logger.info("[Nucleus] Núcleo detenido")
        except Exception as e:
            self.logger.error(f"[Nucleus] Error deteniendo núcleo: {e}")
