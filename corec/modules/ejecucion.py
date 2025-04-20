import logging
import random
from corec.core import ComponenteBase


class ModuloEjecucion(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloEjecucion")
        self.nucleus = None

    async def inicializar(self, nucleus):
        """Inicializa el módulo de ejecución."""
        self.nucleus = nucleus
        self.logger.info("[Ejecucion] Inicializado")

    async def ejecutar(self):
        """Encola tareas para procesar bloques."""
        try:
            registro = self.nucleus.modules.get("registro")
            if not registro:
                self.logger.error("[Ejecucion] Módulo registro no encontrado")
                await self.nucleus.publicar_alerta({
                    "tipo": "error_ejecucion",
                    "mensaje": "Módulo registro no encontrado",
                    "timestamp": random.random()
                })
                return
            for bloque_id, bloque in registro.bloques.items():
                self.logger.info(f"[Ejecucion] Encolando tarea para bloque {bloque_id}")
                try:
                    self.ejecutar_bloque_task.delay(bloque_id)
                    await self.nucleus.publicar_alerta({
                        "tipo": "tarea_encolada",
                        "bloque_id": bloque_id,
                        "timestamp": random.random()
                    })
                except Exception as e:
                    self.logger.error(f"[Ejecucion] Error encolando tarea para {bloque_id}: {e}")
                    await self.nucleus.publicar_alerta({
                        "tipo": "error_tarea",
                        "bloque_id": bloque_id,
                        "mensaje": str(e),
                        "timestamp": random.random()
                    })
        except Exception as e:
            self.logger.error(f"[Ejecucion] Error encolando tareas: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_ejecucion",
                "mensaje": str(e),
                "timestamp": random.random()
            })

    def ejecutar_bloque_task(self, bloque_id: str):
        """Tarea para procesar un bloque (simulada)."""
        self.logger.info(f"[Ejecucion] Procesando bloque {bloque_id}")
        # Simulación de procesamiento
        pass

    async def detener(self):
        """Detiene el módulo de ejecución."""
        self.logger.info("[Ejecucion] Detenido")
