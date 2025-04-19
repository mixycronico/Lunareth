#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/utils/helpers.py
Utilidades para el plugin Codex.
"""
import time
import logging

class CircuitBreaker:
    def __init__(self, max_failures: int, reset_timeout: int):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.logger = logging.getLogger("CircuitBreaker")

    def check(self) -> bool:
        if self.failure_count >= self.max_failures:
            if time.time() - self.last_failure_time < self.reset_timeout:
                self.logger.warning("Circuit breaker abierto")
                return False
            self.failure_count = 0
        return True

    def register_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.logger.error(f"Fallo registrado, conteo: {self.failure_count}")