# corec/entrelazador.py
import logging
import time
import networkx as nx
from typing import Set
from corec.utils.quantization import escalar
from corec.config import MAX_ENLACES_POR_ENTIDAD, REDIS_STREAM_KEY
from corec.entities import EntidadBase

class Entrelazador:
    def __init__(self, redis_client=None):
        """
        Gestiona relaciones entre entidades usando un grafo dirigido.

        Args:
            redis_client: Cliente Redis para persistencia (opcional).
        """
        self.logger = logging.getLogger("Entrelazador")
        self.grafo = nx.DiGraph()
        self.redis_client = redis_client
        self.max_enlaces = MAX_ENLACES_POR_ENTIDAD
        self.entidades: Dict[str, EntidadBase] = {}  # Mapa de ID a entidad

    def registrar_entidad(self, entidad: EntidadBase):
        """Registra una entidad en el grafo."""
        self.entidades[entidad.id] = entidad
        self.grafo.add_node(entidad.id)

    def enlazar(self, entidad_a: EntidadBase, entidad_b: EntidadBase):
        """
        Enlaza dos entidades y persiste el enlace en Redis.

        Args:
            entidad_a (EntidadBase): Primera entidad.
            entidad_b (EntidadBase): Segunda entidad.

        Raises:
            ValueError: Si se excede el límite de enlaces.
        """
        if entidad_a.id not in self.entidades or entidad_b.id not in self.entidades:
            raise ValueError("Ambas entidades deben estar registradas")
        if self.grafo.degree(entidad_a.id) >= self.max_enlaces or self.grafo.degree(entidad_b.id) >= self.max_enlaces:
            self.logger.warning(f"Límite de enlaces alcanzado para {entidad_a.id} o {entidad_b.id}")
            raise ValueError("Límite de enlaces alcanzado")
        self.grafo.add_edge(entidad_a.id, entidad_b.id)
        if self.redis_client:
            try:
                enlace = {
                    "entidad_a": entidad_a.id,
                    "entidad_b": entidad_b.id,
                    "timestamp": time.time()
                }
                self.redis_client.xadd(REDIS_STREAM_KEY, enlace)
                self.logger.info(f"Enlace {entidad_a.id} -> {entidad_b.id} persistido en Redis")
            except Exception as e:
                self.logger.error(f"Error persistiendo enlace: {e}")

    async def afectar(self, entidad: EntidadBase, cambio: Dict[str, float], max_saltos: int = 1):
        """
        Propaga un cambio a entidades enlazadas.

        Args:
            entidad (EntidadBase): Entidad que inicia el cambio.
            cambio (Dict[str, float]): Cambios a propagar.
            max_saltos (int): Máximo número de saltos en el grafo.
        """
        cambio_escalado = {k: escalar(v) for k, v in cambio.items()}
        visited: Set[str] = set()

        async def propagar(nodo_id: str, saltos: int):
            if nodo_id in visited or saltos > max_saltos:
                return
            visited.add(nodo_id)
            entidad_obj = self.entidades.get(nodo_id)
            if entidad_obj and hasattr(entidad_obj, "recibir_cambio"):
                try:
                    entidad_obj.recibir_cambio(cambio_escalado)
                    self.logger.debug(f"Cambio propagado a {nodo_id}")
                except Exception as e:
                    self.logger.error(f"Error propagando cambio a {nodo_id}: {e}")
            for vecino in self.grafo.successors(nodo_id):
                await propagar(vecino, saltos + 1)

        if entidad.id in self.entidades:
            await propagar(entidad.id, 0)
        else:
            self.logger.warning(f"Entidad {entidad.id} no registrada en Entrelazador")
