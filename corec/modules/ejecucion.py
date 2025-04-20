import logging
import random
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico


class ModuloEjecucion(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloEjecucion")
        self.nucleus = None

    async def inicializar(self, nucleus, config=None):
        """Inicializa el módulo de ejecución."""
        try:
            self.nucleus = nucleus
            self.logger.info("[Ejecución] Módulo inicializado")
        except Exception as e:
            self.logger.error(f"[Ejecución] Error inesperado al inicializar: {e}")

    async def encolar_bloque(self, bloque: BloqueSimbiotico):
        """Encola un bloque para procesamiento."""
        try:
            await bloque.procesar(random.uniform(0, 1))
            await self.nucleus.publicar_alerta({
                "tipo": "tarea_encolada",
                "bloque_id": bloque.id,
                "timestamp": random.random()
            })
            self.logger.info(f"[Ejecución] Tarea encolada para bloque {bloque.id}")
        except Exception as e:
            self.logger.error(f"[Ejecución] Error encolando tarea para bloque {bloque.id}: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_encolado",
                "bloque_id": bloque.id,
                "mensaje": str(e),
                "timestamp": random.random()
            })
            raise

    async def detener(self):
        """Detiene el módulo de ejecución."""
        self.logger.info("[Ejecución] Módulo detenido")
