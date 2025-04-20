import logging
import time
from corec.core import ComponenteBase
from corec.celery import celery_app


class ModuloEjecucion(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloEjecucion")
        self.nucleus = None

    async def inicializar(self, nucleus):
        """Inicializa el módulo de ejecución."""
        self.nucleus = nucleus
        self.logger.info("[Ejecucion] Inicializado")

    @celery_app.task
    def ejecutar_bloque_task(self, bloque_id: str, db_config: dict):
        """Tarea Celery para ejecutar un bloque."""
        try:
            bloque = self.nucleus.modules["registro"].bloques.get(bloque_id)
            if bloque:
                bloque.procesar(0.5)  # Carga simulada
                self.logger.info(f"[Ejecucion] Bloque '{bloque_id}' ejecutado")
            else:
                self.logger.error(f"[Ejecucion] Bloque '{bloque_id}' no encontrado")
        except Exception as e:
            self.logger.error(f"[Ejecucion] Error ejecutando bloque '{bloque_id}': {e}")

    async def ejecutar(self):
        """Ejecuta bloques con prioridad para plugins activos."""
        try:
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
            self.logger.error(f"[Ejecucion] Error: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_ejecucion",
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def detener(self):
        """Detiene el módulo de ejecución."""
        self.logger.info("[Ejecucion] Detenido")
