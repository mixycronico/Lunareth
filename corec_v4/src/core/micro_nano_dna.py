import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any
from ..utils.logging import logger
from .neural_nets import NanoNeuralNet

class MicroNanoDNA:
    def __init__(self, funcion_id: str, parametros: Dict[str, Any], mutacion_prob: float = 0.01, fitness: float = 0.0):
        self.funcion_id = funcion_id
        self.parametros = self._validar_parametros(parametros)
        self.mutacion_prob = mutacion_prob
        self.fitness = fitness
        self.neural_net = NanoNeuralNet()
        self.optimizer = optim.Adam(self.neural_net.parameters(), lr=0.005, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        self.logger = logger.getLogger(f"MicroNanoDNA-{funcion_id}")

    def _validar_parametros(self, parametros: Dict[str, Any]) -> Dict[str, Any]:
        validos = {}
        for key, value in parametros.items():
            if isinstance(value, (int, float)):
                validos[key] = max(min(value, 1e6), -1e6)
            else:
                validos[key] = value
        return validos

    def mutar(self):
        if random.random() < self.mutacion_prob and self.fitness < 0.5:
            for key in self.parametros:
                if isinstance(self.parametros[key], (int, float)):
                    factor = 1.03 if self.fitness < 0.3 else 0.97
                    nuevo_valor = self.parametros[key] * random.uniform(factor * 0.99, factor * 1.01)
                    self.parametros[key] = max(min(nuevo_valor, 1e6), -1e6)
            for param in self.neural_net.parameters():
                if random.random() < 0.01:
                    param.data += torch.randn_like(param) * 0.001
        return self

    def recombinar(self, otro: 'MicroNanoDNA') -> 'MicroNanoDNA':
        nuevos_parametros = {}
        for key in self.parametros:
            nuevos_parametros[key] = random.choice([self.parametros[key], otro.parametros[key]])
        nuevo_dna = MicroNanoDNA(self.funcion_id, nuevos_parametros, self.mutacion_prob)
        for param_self, param_other, param_new in zip(self.neural_net.parameters(), otro.neural_net.parameters(), nuevo_dna.neural_net.parameters()):
            mask = torch.rand_like(param_self) > 0.5
            param_new.data = torch.where(mask, param_self.data, param_other.data)
        nuevo_dna.neural_net._memoria = self.neural_net._memoria[-50:]
        return nuevo_dna.mutar()

    def heredar(self) -> 'MicroNanoDNA':
        nuevo_dna = MicroNanoDNA(self.funcion_id, self.parametros.copy(), self.mutacion_prob, self.fitness)
        nuevo_dna.neural_net._memoria = self.neural_net._memoria[:]
        return nuevo_dna.mutar()

    def entrenar_red(self, inputs: torch.Tensor, targets: torch.Tensor):
        try:
            self.optimizer.zero_grad()
            outputs = self.neural_net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.neural_net.parameters(), max_norm=0.5)
            self.optimizer.step()
            for state, _ in self.neural_net._memoria[-10:]:
                outputs_mem = self.neural_net(state)
                loss_mem = self.criterion(outputs_mem, targets)
                loss_mem.backward()
                self.optimizer.step()
            return loss.item()
        except Exception as e:
            self.logger.error(f"[MicroNanoDNA] Error entrenando red: {e}")
            return float('inf')