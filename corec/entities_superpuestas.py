# corec/entities_superpuestas.py
import random
import logging
from typing import Dict
from corec.utils.quantization import escalar
from corec.entities import EntidadBase
from corec.config import QUANTIZATION_STEP_DEFAULT

class EntidadSuperpuesta(EntidadBase):
    def __init__(
        self,
        id: str,
        roles: Dict[str, float],
        quantization_step: float = QUANTIZATION_STEP_DEFAULT,
        min_fitness: float = 0.3,
        mutation_rate: float = 0.1
    ):
        """
        Entidad con múltiples roles cuantizados que se normalizan.

        Args:
            id (str): Identificador único.
            roles (Dict[str, float]): Roles iniciales y sus pesos.
            quantization_step (float): Paso de cuantización específico.
            min_fitness (float): Umbral de fitness para desencadenar mutaciones.
            mutation_rate (float): Probabilidad de mutar roles si fitness es bajo.

        Raises:
            ValueError: Si los roles están vacíos.
        """
        if not roles:
            raise ValueError("Los roles no pueden estar vacíos")
        self.logger = logging.getLogger("EntidadSuperpuesta")
        self.id = id
        self.quantization_step = quantization_step
        self.min_fitness = min_fitness
        self.mutation_rate = mutation_rate
        self.roles = {k: escalar(v, quantization_step) for k, v in roles.items()}
        self.normalizar_roles()

    async def procesar(self, carga: float) -> Dict[str, Any]:
        """
        Procesa la entidad basado en los roles y la carga.

        Args:
            carga (float): Factor de carga (0.0 a 1.0).

        Returns:
            Dict[str, Any]: Resultado con valor cuantizado y copia de roles.
        """
        valor = sum(v * carga for v in self.roles.values())
        return {
            "valor": escalar(valor, self.quantization_step),
            "roles": self.roles.copy()
        }

    def recibir_cambio(self, cambio: Dict[str, float]):
        """Actualiza roles con valores cuantizados y normaliza."""
        for rol, valor in cambio.items():
            self.ajustar_rol(rol, valor)

    def ajustar_rol(self, rol: str, nuevo_peso: float):
        """Ajusta o añade un rol, cuantizado, y normaliza."""
        self.roles[rol] = escalar(nuevo_peso, self.quantization_step)
        self.normalizar_roles()

    def normalizar_roles(self):
        """
        Normaliza los roles para que la suma de sus valores absolutos sea 1.0.
        Si todos los roles son 0, asigna 0 a todos.
        """
        total = sum(abs(v) for v in self.roles.values())
        if total == 0:
            self.roles = {k: escalar(0.0, self.quantization_step) for k in self.roles}
        else:
            self.roles = {k: escalar(v / total, self.quantization_step) for k, v in self.roles}

    def mutar_roles(self, fitness: float, ml_module=None):
        """Mutar roles si el fitness es bajo, usando ML si está disponible."""
        if fitness < self.min_fitness and random.random() < self.mutation_rate:
            if ml_module:
                ajuste = asyncio.run(ml_module.predecir_ajuste_roles(self))
                if ajuste:
                    self.roles = {k: escalar(v, self.quantization_step) for k, v in ajuste.items()}
                    self.normalizar_roles()
                    self.logger.info(f"[Entidad {self.id}] Roles ajustados por ML: {self.roles}")
                    return
            # Mutación aleatoria si no hay ML
            for rol in self.roles:
                delta = random.uniform(-0.1, 0.1)
                self.roles[rol] = escalar(self.roles[rol] + delta, self.quantization_step)
            self.normalizar_roles()
            self.logger.info(f"[Entidad {self.id}] Roles mutados aleatoriamente debido a fitness bajo: {fitness}")

    def crear_entidad(self, bloque_id: str, canal: int) -> 'EntidadSuperpuesta':
        """Crea una nueva entidad con roles derivados."""
        nuevos_roles = {k: escalar(v + random.uniform(-0.05, 0.05), self.quantization_step) for k, v in self.roles.items()}
        nueva_entidad = EntidadSuperpuesta(
            f"{self.id}_child_{random.randint(0, 1000)}",
            nuevos_roles,
            self.quantization_step,
            self.min_fitness,
            self.mutation_rate
        )
        self.logger.info(f"[Entidad {self.id}] Creó nueva entidad: {nueva_entidad.id}")
        return nueva_entidad
