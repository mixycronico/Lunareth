#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/comm_system/processors/memory.py
Memoria conversacional persistente para CommSystem.
"""
import json
import logging
from typing import List, Tuple, Dict, Any
from corec.core import zstd
from aioredis import Redis

class ChatMemory:
    def __init__(self, redis: Redis, ttl: int, disk_path: str):
        self.logger     = logging.getLogger("ChatMemory")
        self.redis      = redis
        self.ttl        = ttl
        self.disk_path  = disk_path
        self.redis_key  = "comm_system:chat_memory"

    async def load(self) -> List[Tuple[str, str]]:
        # Intenta Redis
        try:
            blob = await self.redis.get(self.redis_key)
            if blob:
                data = zstd.decompress(blob)
                return json.loads(data)
        except Exception as e:
            self.logger.error(f"Error cargando memoria de Redis: {e}")
        # Fallback a disco
        try:
            with open(self.disk_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []

    async def save(self, history: List[Tuple[str, str]]):
        # Guarda en Redis
        try:
            blob = zstd.compress(json.dumps(history).encode())
            await self.redis.set(self.redis_key, blob, ex=self.ttl)
        except Exception as e:
            self.logger.error(f"Error guardando memoria en Redis: {e}")
        # Guarda en disco
        try:
            with open(self.disk_path, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Error guardando memoria en disco: {e}")

    def to_openai(self, history: List[Tuple[str, str]], max_pairs: int = 10) -> List[Dict[str, str]]:
        msgs = []
        for q, a in history[-max_pairs:]:
            msgs.append({"role": "user",      "content": q})
            msgs.append({"role": "assistant", "content": a})
        return msgs

    async def add_pair(self, question: str, answer: str):
        hist = await self.load()
        hist.append((question, answer))
        if len(hist) > 50:
            hist = hist[-50:]
        await self.save(hist)