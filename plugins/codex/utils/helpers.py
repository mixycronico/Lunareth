#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/utils/helpers.py
Utilidades para el plugin Codex.
"""

import time
import logging
import asyncio
from pathlib import Path
from typing import Callable, Any

class CircuitBreaker:
    """Permite detener llamadas tras N errores y auto‑reset tras timeout."""
    def __init__(self, max_failures: int, reset_timeout: int):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.logger = logging.getLogger("CircuitBreaker")

    def check(self) -> bool:
        """Devuelve False si el breaker está abierto, True si está cerrado."""
        if self.failure_count >= self.max_failures:
            if time.time() - self.last_failure_time < self.reset_timeout:
                self.logger.warning("Circuit breaker abierto")
                return False
            # timeout expirado, cerramos breaker
            self.failure_count = 0
        return True

    def register_failure(self):
        """Registra un fallo; si supera max_failures, abre el breaker."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.logger.error(f"Fallo registrado, conteo: {self.failure_count}")

async def run_blocking(func: Callable, *args, **kwargs) -> Any:
    """
    Ejecuta una función bloqueante en un executor para no
    bloquear el event loop de asyncio.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

def sanitize_path(base: Path, user_path: str) -> Path:
    """
    Resuelve user_path y garantiza que quede dentro de base.
    Previene escapes como '../../etc/passwd'.
    """
    p = Path(user_path).resolve()
    base_resolved = base.resolve()
    if base_resolved not in p.parents and base_resolved != p:
        raise ValueError(f"Ruta no permitida: {user_path}")
    return p