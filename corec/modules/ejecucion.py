import logging
import asyncio
import time
from typing import Dict, Any
from corec.core import ModuloBase, celery_app
from corec.blocks import BloqueSimbiotico


class ModuloEjecucion(ModuloBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloEjecucion")
        self.nucleus = None

    async def inicializar(self, nucleus):
        """Inicializa el módulo de ejecución."""
        self.nucleus = nucleus
        self.logger.info("[Ejecucion] listo")

    @celery_app.task(autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 5})
    def ejecutar_bloque_task(bloque_id: str, db_config: Dict[str, Any]):
        """Ejecuta una tarea de bloque en Celery."""
        try:
            bloque = BloqueSimbiotico(bloque_id, 0, [], nucleus=None)
            asyncio.run(bloque.escribir_postgresql(db_config))
            logging.getLogger("ModuloEjecucion").info(f"[Ejecucion] Bloque {bloque_id} ok")
        except Exception as e:
            logging.getLogger("ModuloEjecucion").error(f"[Ejecucion] Error {e}")
            raise

    async def ejecutar(self):
        """Ejecuta bloques con prioridad para plugins activos."""
        while True:
            try:
                # Priorizar bloques de plugins
                plugin_bloques = list(self.nucleus.bloques_plugins.values())
                global_bloques = list(self.nucleus.modules["registro"].bloques.values())
                for bloque in plugin_bloques + global_bloques:
                    self.ejecutar_bloque_task.delay(bloque.id, self.nucleus.db_config)
                self.logger.info("[Ejecucion] Tasks encoladas")
                await self.nucleus.publicar_alerta({
                    "tipo": "bloques_encolados",
                    "bloques": [b.id for b in plugin_bloques + global_bloques],
                    "timestamp": time.time()
                })
            except Exception as e:
                self.logger.error(f"[Ejecucion] Error {e}")
                await self.nucleus.publicar_alerta({
                    "tipo": "error_ejecucion",
                    "mensaje": str(e),
                    "timestamp": time.time()
                })
            await asyncio.sleep(60)  # Ejecutar cada 60 segundos

    async def detener(self):
        """Detiene el módulo de ejecución."""
        self.logger.info("[Ejecucion] detenido")
