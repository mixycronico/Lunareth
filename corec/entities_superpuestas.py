import random
import time
import uuid
import json
from typing import Dict, Optional
from corec.utils.quantization import escalar
from corec.entities import EntidadBase


class EntidadSuperpuesta(EntidadBase):
    def __init__(
        self,
        id: str,
        roles: Dict[str, float],
        quantization_step: float = 0.05,
        min_fitness: float = 0.3,
        mutation_rate: float = 0.1,
        nucleus=None
    ):
        """Entidad con múltiples roles cuantizados que se normalizan.

        Args:
            id (str): Identificador único.
            roles (Dict[str, float]): Roles iniciales y sus pesos.
            quantization_step (float): Paso de cuantización.
            min_fitness (float): Umbral de fitness para mutaciones.
            mutation_rate (float): Probabilidad de mutar roles si fitness es bajo.
            nucleus: Instancia del núcleo de CoreC (opcional).

        Raises:
            ValueError: Si los roles están vacíos.
        """
        if not roles:
            raise ValueError("Los roles no pueden estar vacíos")
        self.nucleus = nucleus
        self.logger = nucleus.logger if nucleus else logging.getLogger("CoreC")
        self.id = id
        self.quantization_step = quantization_step
        self.min_fitness = min_fitness
        self.mutation_rate = mutation_rate
        self.roles = {k: escalar(v, quantization_step) for k, v in roles.items()}
        self.normalizar_roles()

    async def procesar(self, carga: float) -> Dict[str, Any]:
        """Procesa la entidad basado en los roles y la carga.

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
        """Actualiza roles con valores cuantizados y normaliza.

        Args:
            cambio (Dict[str, float]): Cambios a aplicar a los roles.
        """
        for rol, valor in cambio.items():
            self.ajustar_rol(rol, valor)

    def ajustar_rol(self, rol: str, nuevo_peso: float):
        """Ajusta o añade un rol, cuantizado, y normaliza.

        Args:
            rol (str): Nombre del rol.
            nuevo_peso (float): Nuevo peso del rol.
        """
        self.roles[rol] = escalar(nuevo_peso, self.quantization_step)
        self.normalizar_roles()

    def normalizar_roles(self):
        """Normaliza los roles para que la suma de sus valores absolutos sea 1.0."""
        total = sum(abs(v) for v in self.roles.values())
        if total == 0:
            self.roles = {k: escalar(0.0, self.quantization_step) for k in self.roles}
        else:
            self.roles = {k: escalar(v / total, self.quantization_step) for k, v in self.roles}

    async def mutar_roles(self, fitness: float, ml_module=None):
        """Muta roles si el fitness es bajo, usando ML si está disponible.

        Args:
            fitness (float): Valor de fitness actual.
            ml_module: Módulo ML para predicciones (opcional).
        """
        if fitness < self.min_fitness and random.random() < self.mutation_rate:
            if ml_module:
                ajuste = await ml_module.predecir_ajuste_roles(self)
                if ajuste:
                    self.roles = {k: escalar(v, self.quantization_step) for k, v in ajuste.items()}
                    self.normalizar_roles()
                    self.logger.info(f"Entidad {self.id} roles ajustados por ML: {self.roles}")
                    return
            for rol in self.roles:
                delta = random.uniform(-0.1, 0.1)
                self.roles[rol] = escalar(self.roles[rol] + delta, self.quantization_step)
            self.normalizar_roles()
            self.logger.info(f"Entidad {self.id} roles mutados aleatoriamente debido a fitness bajo: {fitness}")

    async def crear_entidad(self, bloque_id: str, canal: int, db_pool=None) -> 'EntidadSuperpuesta':
        """Crea una nueva entidad con roles derivados y la persiste en PostgreSQL.

        Args:
            bloque_id (str): ID del bloque al que pertenece la entidad.
            canal (int): Canal de comunicación.
            db_pool: Pool de conexiones a PostgreSQL (opcional).

        Returns:
            EntidadSuperpuesta: Nueva entidad creada.
        """
        nuevos_roles = {k: escalar(v + random.uniform(-0.05, 0.05), self.quantization_step) for k, v in self.roles.items()}
        nueva_entidad = EntidadSuperpuesta(
            f"{self.id}_child_{uuid.uuid4().hex[:8]}",
            nuevos_roles,
            self.quantization_step,
            self.min_fitness,
            self.mutation_rate,
            self.nucleus
        )
        self.logger.info(f"Entidad {self.id} creó nueva entidad: {nueva_entidad.id}")

        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO entidades (
                            entidad_id, bloque_id, roles, quantization_step, min_fitness, mutation_rate, timestamp
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        nueva_entidad.id,
                        bloque_id,
                        json.dumps(nueva_entidad.roles),
                        nueva_entidad.quantization_step,
                        nueva_entidad.min_fitness,
                        nueva_entidad.mutation_rate,
                        time.time()
                    )
                self.logger.info(f"Entidad {nueva_entidad.id} persistida en PostgreSQL")
            except Exception as e:
                self.logger.error(f"Entidad {nueva_entidad.id} error persistiendo en PostgreSQL: {e}")

        return nueva_entidad
