# corec/modules/ml.py
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import Dict, List
from corec.blocks import BloqueSimbiotico

class ModuloML:
    def __init__(self):
        self.logger = logging.getLogger("ModuloML")
        self.modelos: Dict[str, LinearRegression] = {}  # {entidad_id: modelo}
        self.historial: Dict[str, List[Dict]] = {}  # {entidad_id: [{roles, fitness, timestamp}]}

    async def inicializar(self, nucleus, config):
        self.nucleus = nucleus
        self.logger.info("[ML] Módulo inicializado")

    async def entrenar_modelo(self, entidad: 'EntidadSuperpuesta', fitness: float):
        """Entrena un modelo para predecir ajustes de roles."""
        entidad_id = entidad.id
        if entidad_id not in self.historial:
            self.historial[entidad_id] = []

        self.historial[entidad_id].append({
            "roles": entidad.roles.copy(),
            "fitness": fitness,
            "timestamp": time.time()
        })

        # Mantener solo las últimas 50 entradas
        self.historial[entidad_id] = self.historial[entidad_id][-50:]

        # Entrenar modelo si hay suficiente historial
        if len(self.historial[entidad_id]) >= 10:
            X = [list(h["roles"].values()) for h in self.historial[entidad_id]]
            y = [h["fitness"] for h in self.historial[entidad_id]]
            modelo = LinearRegression()
            modelo.fit(X, y)
            self.modelos[entidad_id] = modelo
            self.logger.info(f"[ML] Modelo entrenado para entidad {entidad_id}")

    async def predecir_ajuste_roles(self, entidad: 'EntidadSuperpuesta') -> Dict[str, float]:
        """Predice un ajuste óptimo de roles usando el modelo entrenado."""
        entidad_id = entidad.id
        if entidad_id not in self.modelos:
            return None  # No hay modelo entrenado
        modelo = self.modelos[entidad_id]
        roles_actuales = list(entidad.roles.values())
        # Generar posibles ajustes (variaciones pequeñas)
        posibles_ajustes = []
        for _ in range(10):
            ajuste = {k: v + random.uniform(-0.1, 0.1) for k, v in entidad.roles.items()}
            total = sum(abs(v) for v in ajuste.values())
            if total > 0:
                ajuste = {k: v / total for k, v in ajuste.items()}
            posibles_ajustes.append(list(ajuste.values()))
        # Predecir fitness para cada ajuste
        fitness_predichos = modelo.predict(posibles_ajustes)
        mejor_ajuste_idx = np.argmax(fitness_predichos)
        mejor_ajuste = posibles_ajustes[mejor_ajuste_idx]
        return dict(zip(entidad.roles.keys(), mejor_ajuste))

    async def detener(self):
        self.logger.info("[ML] Módulo detenido")
