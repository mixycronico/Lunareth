#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/processors/memory.py
Almacena estado de revisiones en Redis.
"""
import hashlib
import logging
from typing import Optional

class CodexMemory:
    def __init__(self, redis_client):
        self.redis_client = redis_client
        self.logger = logging.getLogger("CodexMemory")
        self.ttl = 86400  # 1 día

    async def necesita_revision(self, archivo: str, contenido: str) -> bool:
        try:
            hash_actual = hashlib.sha256(contenido.encode()).hexdigest()
            hash_guardado = await self.redis_client.get(f"codex:{archivo}")
            return hash_guardado != hash_actual.encode() if hash_guardado else True
        except Exception as e:
            self.logger.error(f"Error verificando {archivo}: {e}")
            return True

    async def guardar_revision(self, archivo: str, contenido: str):
        try:
            hash_nuevo = hashlib.sha256(contenido.encode()).hexdigest()
            await self.redis_client.setex(f"codex:{archivo}", self.ttl, hash_nuevo)
        except Exception as e:
            self.logger.error(f"Error guardando {archivo}: {e}")