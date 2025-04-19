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
import numpy as np
import torch
import torch.nn as nn
import aiohttp
from typing import Dict, Any
from sklearn.naive_bayes import GaussianNB
from corec.core import serializar_mensaje, deserializar_mensaje, zstd, aioredis
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
        self.q_table[state_key][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

class ComunicadorInteligente:
    def __init__(self, nucleus, config):
        self.nucleus = nucleus
        self.logger = logging.getLogger("ComunicadorInteligente")
        self.config = config
        self.canal = config.get("canal", 4)
        self.bloque = None
        self.openrouter_api_key = config.get("openrouter_api_key")
        self.nn_model = RedNeuronalLigera()
        self.bayes_model = GaussianNB()
        self.rl_model = QLearningAgent(actions=["responder", "analizar", "optimizar"])
        self.training_log = config.get("training_log", "plugins/comunicador_inteligente/data/training.log")
        self.is_openrouter_available = True
        self.redis_client = None
        self.model_path = {
            "nn": "plugins/comunicador_inteligente/models/nn.pt",
            "bayes": "plugins/comunicador_inteligente/models/bayes.pkl",
            "rl": "plugins/comunicador_inteligente/models/rl.pkl"
        }

    async def inicializar(self):
        # Inicializar Redis
        redis_url = f"redis://{self.nucleus.redis_config['username']}:{self.nucleus.redis_config['password']}@{self.nucleus.redis_config['host']}:{self.nucleus.redis_config['port']}"
        self.redis_client = await aioredis.from_url(redis_url, decode_responses=True)
        self.logger.info("Redis inicializado para ComunicadorInteligente")

        # Cargar o inicializar modelos locales
        try:
            with open(self.model_path["nn"], "rb") as f:
                self.nn_model.load_state_dict(torch.load(f))
            with open(self.model_path["bayes"], "rb") as f:
                self.bayes_model = pickle.load(f)
            with open(self.model_path["rl"], "rb") as f:
                self.rl_model = pickle.load(f)
            self.logger.info("Modelos locales cargados")
        except FileNotFoundError:
            self.logger.info("Modelos locales no encontrados, inicializando nuevos")
            torch.save(self.nn_model.state_dict(), self.model_path["nn"])
            with open(self.model_path["bayes"], "wb") as f:
                pickle.dump(self.bayes_model, f)
            self.rl_model.save(self.model_path["rl"])

        # Crear bloque simbiótico
        entidades = [crear_entidad(f"m{i}", self.canal, self._procesar_mensaje) for i in range(self.config.get("entidades", 100))]
        self.bloque = BloqueSimbiotico(f"comunicador_inteligente", self.canal, entidades, max_size=1024, nucleus=self.nucleus)
        self.nucleus.modulos["registro"].bloques[self.bloque.id] = self.bloque
        self.logger.info(f"Plugin ComunicadorInteligente inicializado con {len(entidades)} entidades")
        self.nucleus.registrar_plugin("comunicador_inteligente", self)

        # Iniciar escucha de mensajes
        asyncio.create_task(self._escuchar_mensajes())

    async def _escuchar_mensajes(self):
        """Escuchar mensajes de usuario en un stream de Redis."""
        stream = self.config.get("redis_stream_input", "user_input_stream")
        while True:
            try:
                mensajes = await self.redis_client.xread({stream: "0-0"}, count=10)
                for _, entries in mensajes:
                    for _, data in entries:
                        mensaje = json.loads(data["data"])
                        resultado = await self._procesar_mensaje(mensaje)
                        # Enviar respuesta al usuario via Redis
                        respuesta = await serializar_mensaje(
                            int(time.time_ns() % 1000000), self.canal, resultado["valor"], True
                        )
                        await self.redis_client.xadd(
                            self.config.get("redis_stream_output", "user_output_stream"),
                            {"data": respuesta, "texto": resultado["texto"]}
                        )
            except Exception as e:
                self.logger.error(f"Error escuchando mensajes: {e}")
            await asyncio.sleep(1)

    async def _procesar_mensaje(self, mensaje: Dict[str, Any]):
        """Procesar mensaje de usuario."""
        texto = mensaje.get("texto", "")
        valor = mensaje.get("valor", random.random())
        state = (valor, len(self.bloque.mensajes))
        action = self.rl_model.choose_action(state)

        if action == "responder" and self.is_openrouter_available:
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {"Authorization": f"Bearer {self.openrouter_api_key}"}
                    data = {
                        "model": "openai/gpt-4o-mini",
                        "prompt": f"Usuario pregunta: {texto}. Responde como experto en CoreC.",
                        "max_tokens": 100
                    }
                    async with session.post(
                        "https://openrouter.ai/api/v1/completions", json=data, headers=headers
                    ) as resp:
                        respuesta = await resp.json()
                        texto_respuesta = respuesta["choices"][0]["text"]
                        with open(self.training_log, "a") as f:
                            f.write(json.dumps({"entrada": mensaje, "salida": texto_respuesta}) + "\n")
                        reward = 1.0
                        return {"valor": hash(texto_respuesta) % 1000 / 1000, "texto": texto_respuesta}
            except Exception as e:
                self.logger.error(f"Error con OpenRouter: {e}")
                self.is_openrouter_available = False
                reward = -1.0
        else:
            # Fallback a IAs locales
            respuesta_local = await self._procesar_local(mensaje)
            reward = 0.5 if respuesta_local["valor"] > 0 else -0.5
            texto_respuesta = respuesta_local["texto"]

        next_state = (respuesta_local["valor"] if action != "responder" else valor, len(self.bloque.mensajes))
        self.rl_model.update(state, action, reward, next_state)
        return {"valor": respuesta_local["valor"] if action != "responder" else valor, "texto": texto_respuesta}

    async def _procesar_local(self, mensaje: Dict[str, Any]) -> Dict[str, Any]:
        """Procesar mensaje con IAs locales."""
        valor = mensaje.get("valor", random.random())
        # Red neuronal
        input_tensor = torch.tensor([valor] * 10, dtype=torch.float32)
        with torch.no_grad():
            prediccion_nn = self.nn_model(input_tensor).numpy()[0]

        # Bayesiano
        try:
            bayes_pred = self.bayes_model.predict([[valor]])[0]
        except:
            bayes_pred = "Sistema operativo, fitness estimado: 0.9"

        # Combinar resultados
        texto = bayes_pred if random.random() > 0.5 else f"Análisis local: {prediccion_nn[0]:.2f}"
        return {"valor": prediccion_nn[0], "texto": texto}

    async def entrenar_local(self):
        """Entrenar IAs locales con datos de training.log."""
        try:
            with open(self.training_log, "r") as f:
                datos = [json.loads(line) for line in f]
            if datos:
                X = [[d["entrada"]["valor"]] for d in datos]
                y = [d["salida"] for d in datos]
                self.bayes_model.fit(X, y)
                # Simular entrenamiento de NN (requiere más datos en producción)
                torch.save(self.nn_model.state_dict(), self.model_path["nn"])
                with open(self.model_path["bayes"], "wb") as f:
                    pickle.dump(self.bayes_model, f)
                self.rl_model.save(self.model_path["rl"])
                self.logger.info("Modelos locales entrenados y guardados")
        except Exception as e:
            self.logger.error(f"Error entrenando localmente: {e}")

    async def ejecutar(self):
        """Ejecutar el plugin, procesando mensajes y entrenando IAs."""
        while True:
            resultado = await self.bloque.procesar(self.config.get("carga", 0.5))
            for msg in resultado["mensajes"]:
                if msg.get("texto"):
                    self.logger.info(f"Respuesta al usuario: {msg['texto']}")
            await self.nucleus.publicar_alerta({
                "tipo": "comunicacion",
                "bloque_id": self.bloque.id,
                "fitness": resultado["fitness"],
                "timestamp": time.time()
            })
            await self.entrenar_local()
            await asyncio.sleep(self.config.get("intervalo", 60))

    async def detener(self):
        """Detener el plugin."""
        self.logger.info("Plugin ComunicadorInteligente detenido")
        if self.redis_client:
            await self.redis_client.close()
        # Guardar modelos
        torch.save(self.nn_model.state_dict(), self.model_path["nn"])
        with open(self.model_path["bayes"], "wb") as f:
            pickle.dump(self.bayes_model, f)
        self.rl_model.save(self.model_path["rl"])

def inicializar(nucleus, config):
    plugin = ComunicadorInteligente(nucleus, config)
    asyncio.create_task(plugin.inicializar())