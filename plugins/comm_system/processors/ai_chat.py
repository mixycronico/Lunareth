#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/comm_system/processors/ai_chat.py
Chat real con OpenRouter (google/gemini‑2.0‑flash‑lite‑001).
"""
import os
import asyncio
import logging
from typing import List, Dict, Any
import openai

class AIChat:
    def __init__(self, config: dict):
        self.logger      = logging.getLogger("AIChat")
        self.model       = config["openrouter_model"]
        self.api_key     = config["openrouter_api_key"]
        self.api_base    = config["openrouter_api_base"]
        self.max_tokens  = config.get("max_tokens", 200)
        self.temperature = config.get("temperature", 0.7)
        openai.api_key      = self.api_key
        openai.api_base     = self.api_base
        openai.api_type     = "openrouter"
        openai.api_version  = None

    async def chat(self, history: List[Dict[str, str]], user_input: str) -> str:
        messages = [{"role": "system",
                     "content": "Eres el asistente inteligente de CoreC, experto en su arquitectura."}]
        messages += history
        messages.append({"role": "user", "content": user_input})

        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"[AIChat] Error OpenRouter: {e}")
            return "Lo siento, hubo un problema accediendo a la IA."