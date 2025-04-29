import asyncio
from corec.utils.db_utils import init_postgresql, init_redis


class ModuloAutosanacion:
    def __init__(self):
        self.nucleus = None

    async def inicializar(self, nucleus, config):
        """Inicializa el módulo de autosanación.

        Args:
            nucleus: Instancia del núcleo de CoreC.
            config: Configuración del módulo (opcional).
        """
        self.nucleus = nucleus
        self.logger = nucleus.logger
        self.logger.info("Módulo Autosanación inicializado")

    async def verificar_estado(self):
        """Verifica el estado de los módulos y conexiones, aplicando autosanación si es necesario."""
        try:
            # Verificar conexiones a PostgreSQL
            if not self.nucleus.db_pool:
                self.logger.warning("PostgreSQL no disponible, intentando reconectar")
                try:
                    self.nucleus.db_pool = await init_postgresql(self.nucleus.config.db_config.model_dump())
                    self.logger.info("PostgreSQL reconectado")
                except Exception as e:
                    self.logger.error(f"Error reconectando PostgreSQL: {e}")
                    await self.nucleus.publicar_alerta({
                        "tipo": "error_reconexion_postgresql",
                        "mensaje": str(e),
                        "timestamp": time.time()
                    })

            # Verificar conexiones a Redis
            if not self.nucleus.redis_client:
                self.logger.warning("Redis no disponible, intentando reconectar")
                try:
                    self.nucleus.redis_client = await init_redis(self.nucleus.config.redis_config.model_dump())
                    self.logger.info("Redis reconectado")
                except Exception as e:
                    self.logger.error(f"Error reconectando Redis: {e}")
                    await self.nucleus.publicar_alerta({
                        "tipo": "error_reconexion_redis",
                        "mensaje": str(e),
                        "timestamp": time.time()
                    })

            # Verificar módulos
            for name, module in self.nucleus.modules.items():
                try:
                    await module.inicializar(self.nucleus, self.nucleus.config.get(f"{name}_config", {}).model_dump())
                    self.logger.debug(f"Módulo {name} operativo")
                except Exception as e:
                    self.logger.warning(f"Módulo {name} falló, reiniciando: {e}")
                    module.__init__()
                    await module.inicializar(self.nucleus, self.nucleus.config.get(f"{name}_config", {}).model_dump())
                    self.logger.info(f"Módulo {name} reiniciado")
                    await self.nucleus.publicar_alerta({
                        "tipo": f"reinicio_modulo_{name}",
                        "mensaje": f"Módulo {name} reiniciado debido a error: {e}",
                        "timestamp": time.time()
                    })

        except Exception as e:
            self.logger.error(f"Error verificando estado: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_autosanacion",
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def detener(self):
        """Detiene el módulo de autosanación."""
        self.logger.info("Módulo Autosanación detenido")
