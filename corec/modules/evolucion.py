import random
import time
from typing import Dict, List
from corec.blocks import BloqueSimbiotico


class ModuloEvolucion:
    def __init__(self):
        self.estrategias = [
            {"quantization_step": 0.1, "max_concurrent_tasks": 100, "increment_factor": 1.05},
            {"quantization_step": 0.05, "max_concurrent_tasks": 50, "increment_factor": 1.03},
            {"quantization_step": 0.01, "max_concurrent_tasks": 20, "increment_factor": 1.1}
        ]
        self.historial: Dict[str, List[Dict]] = {}  # {bloque_id: [{estrategia, fitness, timestamp}]}
        self.nucleus = None

    async def inicializar(self, nucleus, config):
        """Inicializa el módulo de evolución.

        Args:
            nucleus: Instancia del núcleo de CoreC.
            config: Configuración del módulo (opcional).
        """
        self.nucleus = nucleus
        self.logger = nucleus.logger
        self.logger.info("Módulo Evolución inicializado")

    async def evaluar_estrategia(self, bloque: BloqueSimbiotico):
        """Evalúa el rendimiento del bloque y propone una nueva estrategia si es necesario.

        Args:
            bloque (BloqueSimbiotico): Bloque a evaluar.
        """
        bloque_id = bloque.id
        fitness = bloque.fitness
        if bloque_id not in self.historial:
            self.historial[bloque_id] = []

        self.historial[bloque_id].append({
            "estrategia": {
                "quantization_step": bloque.quantization_step,
                "max_concurrent_tasks": bloque.max_concurrent_tasks,
                "increment_factor": bloque.increment_factor
            },
            "fitness": fitness,
            "timestamp": time.time()
        })

        # Mantener solo las últimas 10 entradas
        self.historial[bloque_id] = self.historial[bloque_id][-10:]

        # Si el fitness es bajo, probar una nueva estrategia
        min_fitness = bloque.max_errores
        if fitness < min_fitness:
            nueva_estrategia = random.choice(self.estrategias)
            bloque.quantization_step = nueva_estrategia["quantization_step"]
            bloque.max_concurrent_tasks = nueva_estrategia["max_concurrent_tasks"]
            bloque.increment_factor = nueva_estrategia["increment_factor"]
            self.logger.info(f"Bloque {bloque_id} cambió estrategia: {nueva_estrategia}")
            # Publicar aprendizaje si la estrategia parece prometedora
            if fitness > min_fitness * 1.5:
                await self.nucleus.publicar_aprendizaje({
                    "bloque_id": bloque_id,
                    "estrategia": nueva_estrategia,
                    "fitness": fitness,
                    "timestamp": time.time()
                })

    async def detener(self):
        """Detiene el módulo de evolución."""
        self.logger.info("Módulo Evolución detenido")
