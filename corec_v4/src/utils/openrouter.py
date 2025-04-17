#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/utils/openrouter.py
"""
openrouter.py
Cliente para interactuar con OpenRouter, con caché en Redis para respuestas frecuentes.
"""

import aiohttp
import asyncio
from typing import Dict, Any, Optional
from .logging import logger
from .config import get_openrouter_config
import redis.asyncio as aioredis
import json
import hashlib

class OpenRouterClient:
    def __init__(self):
        self.config = get_openrouter_config()
        self.enabled = self.config.get("enabled", False)
        self.api_key = self.config.get("api_key", "")
        self.endpoint = self.config.get("endpoint", "https://openrouter.ai/api/v1")
        self.default_model = self.config.get("model", "nous-hermes-2")
        self.max_tokens = self.config.get("max_tokens", 1000)
        self.temperature = self.config.get("temperature", 0.7)
        self.session = None
        self.logger = logger.getLogger("OpenRouterClient")
        self.rate_limit_delay = 1.0
        self.redis = aioredis.from_url("redis://redis:6379")

    async def initialize(self):
        if self.enabled:
            try:
                self.session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {self.api_key}"})
                self.logger.info("OpenRouterClient inicializado")
            except Exception as e:
                self.logger.error(f"Error inicializando OpenRouter: {e}")
                self.enabled = False
        else:
            self.logger.info("OpenRouterClient deshabilitado")

    async def close(self):
        if self.session:
            await self.session.close()
        await self.redis.close()
        self.logger.info("OpenRouterClient cerrado")

    async def query(self, prompt: str, model: Optional[str] = None, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> Dict[str, Any]:
        if not self.enabled:
            return self._fallback_query(prompt)

        # Verificar caché
        cache_key = hashlib.sha256(prompt.encode()).hexdigest()
        cached_response = await self.redis.get(f"openrouter:{cache_key}")
        if cached_response:
            self.logger.debug("Respuesta obtenida de caché")
            return json.loads(cached_response)

        try:
            payload = {
                "model": model or self.default_model,
                "prompt": prompt,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature
            }
            async with self.session.post(f"{self.endpoint}/chat/completions", json=payload, timeout=10) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"Error en OpenRouter: {response.status} - {error_text}")
                    return self._fallback_query(prompt)
                result = await response.json()
                response_data = {
                    "estado": "ok",
                    "respuesta": result["choices"][0]["message"]["content"]
                }
                # Almacenar en caché
                await self.redis.setex(f"openrouter:{cache_key}", 3600, json.dumps(response_data))
                return response_data
        except Exception as e:
            self.logger.error(f"Excepción en OpenRouter: {e}")
            return self._fallback_query(prompt)
        finally:
            await asyncio.sleep(self.rate_limit_delay)

    def _fallback_query(self, prompt: str) -> Dict[str, Any]:
        self.logger.warning("Usando fallback para prompt: %s", prompt[:50])
        return {
            "estado": "fallback",
            "respuesta": "No se pudo conectar con OpenRouter. Respuesta básica: datos recibidos."
        }

    async def analyze(self, data: Any, context: str) -> Dict[str, Any]:
        prompt = f"Contexto: {context}\nDatos: {data}\nInstrucciones: Analiza los datos y proporciona insights detallados."
        response = await self.query(prompt)
        if response["estado"] == "fallback":
            try:
                summary = {"summary": f"Datos analizados localmente. Tamaño: {len(str(data))} caracteres."}
                return {"estado": "ok", "respuesta": summary}
            except Exception as e:
                return {"estado": "error", "mensaje": f"Error en análisis local: {e}"}
        return response

    async def chat(self, message: str, context: Optional[str] = None) -> Dict[str, Any]:
        prompt = f"Contexto: {context or 'Sistema CoreC'}\nMensaje: {message}\nInstrucciones: Responde de manera clara y precisa."
        response = await self.query(prompt)
        if response["estado"] == "fallback":
            return {"estado": "ok", "respuesta": "CoreC está operativo. No se pudo conectar con IA externa."}
        return response