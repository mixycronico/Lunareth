import time
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Optional
from corec.entities_superpuestas import EntidadSuperpuesta


class ModuloML:
    def __init__(self):
        self.modelos: Dict[str, LinearRegression] = {}  # {entidad_id: modelo}
        self.historial: Dict[str, List[Dict]] = {}  # {entidad_id: [{roles, fitness, timestamp}]}
        self.nucleus = None

    async def inicializar(self, nucleus, config):
        """Inicializa el módulo de aprendizaje automático.

        Args:
            nucleus: Instancia del núcleo de CoreC.
            config: Configuración del módulo ML.
        """
        self.nucleus = nucleus
        self.logger = nucleus.logger
        self.logger.info("Módulo ML inicializado")

    async def entrenar_modelo(self, entidad: EntidadSuperpuesta, fitness: float):
        """Entrena un modelo para predecir ajustes de roles.

        Args:
            entidad (EntidadSuperpuesta): Entidad para entrenar el modelo.
            fitness (float): Valor de fitness asociado.
        """
        entidad_id = entidad.id
        if entidad_id not in self.historial:
            self.historial[entidad_id] = []

        self.historial[entidad_id].append({
            "roles": entidad.roles.copy(),
            "fitness": fitness,
            "timestamp": time.time()
        })

        # Mantener solo las últimas entradas según historial_size
        historial_size = self.nucleus.config.ml_config.historial_size
        self.historial[entidad_id] = self.historial[entidad_id][-historial_size:]

        # Entrenar modelo si hay suficiente historial
        min_samples = self.nucleus.config.ml_config.min_samples_train
        if len(self.historial[entidad_id]) >= min_samples:
            X = [list(h["roles"].values()) for h in self.historial[entidad_id]]
            y = [h["fitness"] for h in self.historial[entidad_id]]
            modelo = LinearRegression()
            modelo.fit(X, y)
            self.modelos[entidad_id] = modelo
            self.logger.info(f"Modelo entrenado para entidad {entidad_id}")

    async def predecir_ajuste_roles(self, entidad: EntidadSuperpuesta) -> Optional[Dict[str, float]]:
        """Predice un ajuste óptimo de roles usando el modelo entrenado.

        Args:
            entidad (EntidadSuperpuesta): Entidad para ajustar roles.

        Returns:
            Optional[Dict[str, float]]: Ajuste de roles predicho o None si no hay modelo.
        """
        entidad_id = entidad.id
        if entidad_id not in self.modelos:
            self.logger.debug(f"No hay modelo entrenado para entidad {entidad_id}")
            return None
        modelo = self.modelos[entidad_id]
        posibles_ajustes = []
        for _ in range(10):
            ajuste = {k: v + random.uniform(-0.1, 0.1) for k, v in entidad.roles.items()}
            total = sum(abs(v) for v in ajuste.values())
            if total > 0:
                ajuste = {k: v / total for k, v in ajuste.items()}
            posibles_ajustes.append(list(ajuste.values()))
        fitness_predichos = modelo.predict(posibles_ajustes)
        mejor_ajuste_idx = np.argmax(fitness_predichos)
        mejor_ajuste = posibles_ajustes[mejor_ajuste_idx]
        return dict(zip(entidad.roles.keys(), mejor_ajuste))

    async def detener(self):
        """Detiene el módulo de aprendizaje automático."""
        self.logger.info("Módulo ML detenido")
