#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/processors/manager.py
Gestiona optimización, generación de websites y plugins en Codex.
"""
import os
import asyncio
import logging
from corec.core import ComponenteBase, zstd, serializar_mensaje
from plugins.codex.processors.reviser import CodexReviser
from plugins.codex.processors.generator import CodexGenerator
from plugins.codex.processors.memory import CodexMemory
from plugins.codex.utils.helpers import CircuitBreaker
import fnmatch
from typing import Dict, Any

class CodexManager(ComponenteBase):
    def __init__(self, nucleus, config):
        super().__init__()
        self.nucleus = nucleus
        self.config = config
        self.logger = logging.getLogger("CodexManager")
        self.directorio = config.get("directorio_objetivo", "plugins/")
        self.exclude_patterns = config.get("exclude_patterns", [])
        self.max_file_size = config.get("max_file_size", 1000000)
        self.reviser = CodexReviser(config)
        self.generator = CodexGenerator(config)
        self.memory = CodexMemory(nucleus.redis_client)
        self.circuit_breaker = CircuitBreaker(
            config.get("circuit_breaker", {}).get("max_failures", 3),
            config.get("circuit_breaker", {}).get("reset_timeout", 900)
        )

    async def inicializar(self):
        os.makedirs("plugins/codex/logs", exist_ok=True)
        os.makedirs(self.config.get("website_output_dir", "generated_websites/"), exist_ok=True)
        os.makedirs(self.config.get("plugin_output_dir", "plugins/"), exist_ok=True)
        self.logger.info("Inicializando CodexManager...")

    async def ejecutar(self):
        while True:
            if not self.circuit_breaker.check():
                await asyncio.sleep(60)
                continue
            try:
                archivos = await self._buscar_archivos()
                for ruta in archivos:
                    if not any(ruta.endswith(ext) for ext in [".py", ".js"]):
                        continue
                    if os.path.getsize(ruta) > self.max_file_size:
                        self.logger.warning(f"Archivo {ruta} excede max_file_size")
                        continue
                    codigo = await self._leer_archivo(ruta)
                    if await self.memory.necesita_revision(ruta, codigo):
                        nuevo = await self.reviser.revisar_codigo(codigo, ruta)
                        if nuevo and nuevo != codigo:
                            await self._escribir_archivo(ruta, nuevo)
                            await self.memory.guardar_revision(ruta, nuevo)
                            await self.redis_client.xadd("corec_commands", {
                                "comando": f"codex revised {ruta}",
                                "data": zstd.compress(json.dumps({"ruta": ruta}).encode())
                            })
            except Exception as e:
                self.logger.error(f"Error en ejecución: {e}")
                self.circuit_breaker.register_failure()
            await asyncio.sleep(self.config.get("intervalo_revision", 300))

    async def manejar_comando(self, comando: Dict[str, Any]) -> Dict[str, Any]:
        try:
            action = comando.get("action")
            params = comando.get("params", {})
            if action == "revise":
                ruta = params.get("file")
                codigo = await self._leer_archivo(ruta)
                nuevo = await self.reviser.revisar_codigo(codigo, ruta)
                if nuevo:
                    await self._escribir_archivo(ruta, nuevo)
                    await self.memory.guardar_revision(ruta, nuevo)
                    await self.redis_client.xadd("corec_commands", {
                        "comando": f"codex revised {ruta}"
                    })
                    return {"status": "ok", "file": ruta}
            elif action == "generate_website":
                result = await self.generator.generar_website(params)
                await self.redis_client.xadd("corec_commands", {
                    "comando": f"codex website_generated {result['output_dir']}"
                })
                return result
            elif action == "generate_plugin":
                result = await self.generator.generar_plugin(params)
                await self.redis_client.xadd("corec_commands", {
                    "comando": f"codex plugin_generated {result['output_dir']}"
                })
                return result
            return {"status": "error", "message": "Acción no soportada"}
        except Exception as e:
            self.logger.error(f"Error en comando: {e}")
            return {"status": "error", "message": str(e)}

    async def _buscar_archivos(self):
        archivos = []
        for root, _, files in os.walk(self.directorio):
            for file in files:
                ruta = os.path.join(root, file)
                if not any(fnmatch.fnmatch(ruta, pattern) for pattern in self.exclude_patterns):
                    archivos.append(ruta)
        return archivos

    async def _leer_archivo(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error leyendo {path}: {e}")
            return ""

    async def _escribir_archivo(self, path: str, contenido: str):
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(contenido)
        except Exception as e:
            self.logger.error(f"Error escribiendo {path}: {e}")
            self.circuit_breaker.register_failure()