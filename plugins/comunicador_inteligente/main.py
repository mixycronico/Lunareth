#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/comunicador_inteligente/main.py
Plugin que añade comunicación con el usuario y razonamiento avanzado a CoreC.
"""

import asyncio
import logging
import json
import time
import random
import pickle
import torch
import torch.nn as nn
import aiohttp
from typing import Dict, Any
from sklearn.naive_bayes import GaussianNB
from corec.core import serializar_mensaje, aioredis
from corec.entities import crear_entidad
from corec.blocks import BloqueSimbiotico


class RedNeuronalLigera(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)


class QLearningAgent:
    def __init__(self, actions: list, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state_key(self, state: tuple) -> str:
        return str(state)

    def choose_action(self, state: tuple) -> str:
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.actions}
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state_key], key=self.q_table[state_key].get)

    def update(self, state: tuple, action: str, reward: float, next_state: tuple):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.actions}
        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values())
        self.q_table[state_key][action] = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
