import time
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico


class ModuloEjecucion(ComponenteBase):
    def __init__(self):
        self.nucleus = None

    async def inicializar(self, nucleus, config=None):
        """Inicializa el módulo de ejecución.

        Args:
            nucleus: Instancia del núcleo de CoreC.
            config: Configuración del módulo (opcional).
        """
        try:
            self.nucleus = nucleus
            self.logger = nucleus.logger
            self.logger.info("Módulo Ejecución inicializado")
        except Exception as e:
            self.logger.error(f"Error inicializando Módulo Ejecución: {e}")
            raise

    async def encolar_bloque(self, bloque: BloqueSimbiotico):
        """Encola un bloque para procesamiento.

        Args:
            bloque (BloqueSimbiotico): Bloque a procesar.
        """
        try:
            import random  # Mantengo random para la carga, pero no para timestamps
            await bloque.procesar(random.uniform(0, 1))
            await self.nucleus.publicar_alerta({
                "tipo": "tarea_encolada",
                "bloque_id": bloque.id,
                "timestamp": time.time()
            })
            self.logger.info(f"Tarea encolada para bloque {bloque.id}")
        except Exception as e:
            self.logger.error(f"Error encolando tarea para bloque {bloque.id}: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_encolado",
                "bloque_id": bloque.id,
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def detener(self):
        """Detiene el módulo de ejecución."""
        self.logger.info("Módulo Ejecución detenido")
