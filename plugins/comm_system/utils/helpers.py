#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/comm_system/utils/helpers.py
Funciones auxiliares para filesystem, I/O asíncrono, validación, etc.
"""
from pathlib import Path
import asyncio

async def run_blocking(func, *args, **kwargs):
    """Ejecuta func en executor para no bloquear asyncio."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))