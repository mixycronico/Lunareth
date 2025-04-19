#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/utils/helpers.py
Funciones auxiliares para el plugin CryptoTrading.
"""
import logging
from datetime import datetime, timedelta

class CircuitBreaker:
    def __init__(self, max_failures: int, reset_timeout: int):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.tripped = False
        self.reset_time = None
        self.logger = logging.getLogger("CircuitBreaker")

    def check(self) -> bool:
        if self.tripped:
            now = datetime.utcnow()
            if now >= self.reset_time:
                self.tripped = False
                self.failure_count = 0
                self.reset_time = None
                self.logger.info("Circuit breaker reseteado")
            else:
                self.logger.warning(f"Circuit breaker activo hasta {self.reset_time}")
                return False
        return True

    def register_failure(self):
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.tripped = True
            self.reset_time = datetime.utcnow() + timedelta(seconds=self.reset_timeout)
            self.logger.error(f"Circuit breaker activado hasta {self.reset_time}")