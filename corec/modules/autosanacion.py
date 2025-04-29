# corec/modules/autosanacion.py
import logging
import asyncio
from corec.utils.db_utils import init_postgresql, init_redis

class ModuloAutosanacion:
    def __init__(self):
        self.logger = logging.getLogger("ModuloAutosanacion")
        self.nucleus = None

    async def inicializar(self, nucleus, config):
        self.nucleus = nucleus
        self.logger.info("[Autosanacion] Módulo inicializado")

    async def verificar_estado(self):
        """Verifica el estado de los módulos y conexiones, aplicando autosanación si es necesario."""
        try:
            # Verificar conexiones a PostgreSQL
            if not self.nucleus.db_pool:
                self.logger.warning("[Autosanacion] PostgreSQL no disponible, intentando reconectar")
                try:
                    self.nucleus.db_pool = await init_postgresql(self.nucleus.config.get("db_config", {}))
                    self.logger.info("[Autosanacion] PostgreSQL reconectado")
                except Exception as e:
                    self.logger.error(f"[Autosanacion] Error reconectando PostgreSQL: {e}")

            # Verificar conexiones a Redis
            if not self.nucleus.redis_client:
                self.logger.warning("[Autosanacion] Redis no disponible, intentando reconectar")
                try:
                    self.nucleus.redis_client = await init_redis(self.nucleus.config.get("redis_config", {}))
                    self.logger.info("[Autosanacion] Redis reconectado")
                except Exception as e:
                    self.logger.error(f"[Autosanacion] Error reconectando Redis: {e}")

            # Verificar módulos
            for name, module in self.nucleus.modules.items():
                try:
                    # Simular una operación simple para verificar el módulo
                    await module.inicializar(self.nucleus, self.nucleus.config.get(f"{name}_config", {}))
                    self.logger.debug(f"[Autosanacion] Módulo {name} operativo")
                except Exception as e:
                    self.logger.warning(f"[Autosanacion] Módulo {name} falló, reiniciando: {e}")
                    module.__init__()  # Reiniciar módulo
                    await module.inicializar(self.nucleus, self.nucleus.config.get(f"{name}_config", {}))

        except Exception as e:
            self.logger.error(f"[Autosanacion] Error verificando estado: {e}")

    async def detener(self):
        self.logger.info("[Autosanacion] Módulo detenido")
