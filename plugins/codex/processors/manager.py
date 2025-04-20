#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/processors/manager.py
Despacha acciones codex: generate_plugin, generate_website, revise.
Incluye métricas Prometheus, validación y sanitización.
"""
import asyncio
import json
import logging
from pathlib import Path
from prometheus_client import Counter, Histogram, start_http_server
from pydantic import ValidationError
from corec.core import aioredis
from typing import Dict, Any

from .schemas import Cmd, CmdGeneratePlugin, CmdGenerateWebsite, CmdRevise
from .generator import Generator
from .reviser import CodexReviser
from .utils.helpers import run_blocking, sanitize_path

# Métricas
CMD_COUNTER   = Counter("codex_commands_total",   "Comandos recibidos", ["action"])
ERROR_COUNTER = Counter("codex_errors_total",     "Errores internos",   ["action"])
LATENCY_HIST  = Histogram("codex_latency_seconds","Latencia por acción", ["action"])

class CodexManager:
    def __init__(self, nucleus, config: Dict[str,Any]):
        self.nucleus    = nucleus
        self.config     = config
        self.logger     = logging.getLogger("CodexManager")
        self.redis      = None
        self.generator  = Generator(config)
        self.reviser    = CodexReviser(config)
        # Carpeta base para seguridad
        self.base_dir   = Path(config["templates_dir"]).parent.resolve()
        # Streams
        self.stream_in  = config.get("stream_in",  "corec_commands")
        self.stream_out = config.get("stream_out", "corec_responses")
        # Inicia servidor de métricas
        start_http_server(self.config.get("metrics_port", 8001))

    async def init(self):
        url = (
            f"redis://{self.nucleus.redis_config['username']}:"
            f"{self.nucleus.redis_config['password']}@"
            f"{self.nucleus.redis_config['host']}:"
            f"{self.nucleus.redis_config['port']}"
        )
        self.redis = await aioredis.from_url(url, decode_responses=True)

    async def run_loop(self):
        last_id = "0-0"
        while True:
            entries = await self.redis.xread({self.stream_in: last_id}, count=1, block=5000)
            if not entries:
                continue
            _, msgs = entries[0]
            for msg_id, fields in msgs:
                raw = fields.get("data")
                try:
                    # Validación de esquema
                    cmd_obj = Cmd.parse_raw(raw)
                except ValidationError as e:
                    resp = {"status":"error","message":"Payload inválido","details": e.errors()}
                else:
                    action = cmd_obj.action
                    CMD_COUNTER.labels(action=action).inc()
                    with LATENCY_HIST.labels(action=action).time():
                        try:
                            resp = await self.handle(cmd_obj)
                        except Exception as e:
                            ERROR_COUNTER.labels(action=action).inc()
                            self.logger.error(f"Error manejando {action}: {e}")
                            resp = {"status":"error","message": str(e)}
                # Publica respuesta
                await self.redis.xadd(self.stream_out, {"data": json.dumps(resp)})
                last_id = msg_id

    async def handle(self, cmd: Cmd) -> Dict[str,Any]:
        if isinstance(cmd, CmdGeneratePlugin):
            name = cmd.params.plugin_name
            return await self.generator.generate_plugin({"plugin_name": name})

        if isinstance(cmd, CmdGenerateWebsite):
            return await self.generator.generate_website(cmd.params.dict())

        if isinstance(cmd, CmdRevise):
            # Sanitiza ruta dentro del proyecto
            p = sanitize_path(self.base_dir, cmd.params.file)
            # Lee contenido de forma no bloqueante
            src = await run_blocking(p.read_text, encoding="utf-8")
            nuevo = await self.reviser.revisar_codigo(src, str(p))
            if nuevo and nuevo != src:
                await run_blocking(p.write_text, nuevo, encoding="utf-8")
                return {"status":"ok","message":"Archivo revisado","path": str(p)}
            return {"status":"ok","message":"Sin cambios","path": str(p)}

        # Acción no soportada
        return {"status":"error","message":f"Acción '{cmd.action}' no soportada"}

    async def teardown(self):
        # Cierra recursos si fuera necesario
        if self.redis:
            await self.redis.close()