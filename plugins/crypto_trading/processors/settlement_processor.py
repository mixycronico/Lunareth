#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/utils/helpers.py

Funciones auxiliares para el plugin CryptoTrading:
- CircuitBreaker para tolerancia a fallos
- Funciones asincrónicas de utilidad
- Seguridad en acceso a rutas
"""

import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Any


class CircuitBreaker:
    """
    Mecanismo de protección contra errores repetidos.
    Si supera `max_failures`, se activa durante `reset_timeout` segundos.
    """
    def __init__(self, max_failures: int, reset_timeout: int):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.tripped = False
        self.reset_time = None
        self.logger = logging.getLogger("CircuitBreaker")

    def check(self) -> bool:
        """
        Verifica si el circuito está cerrado (activo). Si está abierto, revisa si ya se puede resetear.
        """
        if self.tripped:
            now = datetime.utcnow()
            if now >= self.reset_time:
                self.tripped = False
                self.failure_count = 0
                self.reset_time = None
                self.logger.info("Circuit breaker reseteado automáticamente.")
            else:
                self.logger.warning(f"Circuit breaker activo hasta {self.reset_time}")
                return False
        return True

    def register_failure(self):
        """
        Registra un fallo. Si se supera el máximo, activa el circuito.
        """
        self.failure_count += 1
        self.logger.warning(f"Fallo #{self.failure_count} registrado")
        if self.failure_count >= self.max_failures:
            self.tripped = True
            self.reset_time = datetime.utcnow() + timedelta(seconds=self.reset_timeout)
            self.logger.error(f"Circuit breaker ACTIVADO hasta {self.reset_time}")


async def run_blocking(func: Callable, *args, **kwargs) -> Any:
    """
    Ejecuta una función de bloqueo (como copytree) de forma asíncrona.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def sanitize_path(base: Path, user_path: str) -> Path:
    """
    Asegura que una ruta proporcionada por el usuario esté dentro del directorio base permitido.
    Lanza ValueError si intenta salir del sandbox.
    """
    p = Path(user_path).resolve()
    if base.resolve() not in p.parents and base.resolve() != p:
        raise ValueError("Ruta no permitida")
    return p