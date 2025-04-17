import asyncio
from ..utils.logging import logger
from .base import ModuloBase

class ModuloEjecucion(ModuloBase):
    def __init__(self):
        self.logger = logger.getLogger("ModuloEjecucion")
        self.tasks = []

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.logger.info("[ModuloEjecucion] Inicializado")

    async def ejecutar(self):
        try:
            self.tasks = [
                *(celu.ejecutar() for celu in self.nucleus.modulo_registro.celu_entidades.values()),
                *(micro.ejecutar() for micro in self.nucleus.modulo_registro.micro_celu_entidades.values())
            ]
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except Exception as e:
            self.logger.error(f"[ModuloEjecucion] Error en ejecución: {e}")

    async def detener(self):
        for task in self.tasks:
            task.cancel()
        self.logger.info("[ModuloEjecucion] Detenido")