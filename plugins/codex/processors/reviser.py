#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/processors/reviser.py
Optimiza código existente en el plugin Codex.
"""
import ast
import black
import pyflakes.api
from io import StringIO
from transformers import pipeline
import logging
from typing import Optional

class CodexReviser:
    def __init__(self, config):
        self.logger = logging.getLogger("CodexReviser")
        self.languages = config.get("languages", ["python"])
        try:
            self.ai_pipeline = pipeline("code-generation", model="Salesforce/codet5-small", device=-1)
        except Exception as e:
            self.logger.warning(f"Error cargando CodeT5: {e}, usando herramientas estáticas")
            self.ai_pipeline = None

    async def revisar_codigo(self, codigo: str, archivo: str) -> Optional[str]:
        try:
            if archivo.endswith(".py"):
                return await self._revisar_python(codigo)
            elif archivo.endswith(".js"):
                return await self._revisar_javascript(codigo)
            return codigo
        except Exception as e:
            self.logger.error(f"Error revisando {archivo}: {e}")
            return None

    async def _revisar_python(self, codigo: str) -> str:
        # Análisis estático con pyflakes
        reporter = StringIO()
        pyflakes.api.check(codigo, "<string>", reporter)
        if reporter.getvalue():
            self.logger.warning(f"Errores pyflakes: {reporter.getvalue()}")

        # Refactorización con ast
        try:
            tree = ast.parse(codigo)
            for node in ast.walk(tree):
                if isinstance(node, ast.Pass) and hasattr(node, "parent") and node.parent.body:
                    node.parent.body.remove(node)
            codigo = ast.unparse(tree)
        except SyntaxError:
            self.logger.warning("Error de sintaxis, omitiendo ast")

        # Formato con black
        try:
            codigo = black.format_str(codigo, mode=black.FileMode())
        except Exception as e:
            self.logger.warning(f"Error en black: {e}")

        # Sugerencias AI (CodeT5)
        if self.ai_pipeline:
            try:
                sugerencia = self.ai_pipeline(codigo, max_length=512)[0]["generated_text"]
                if sugerencia != codigo:
                    self.logger.info("Aplicando sugerencia AI")
                    return sugerencia
            except Exception as e:
                self.logger.warning(f"Error en AI: {e}")

        return codigo

    async def _revisar_javascript(self, codigo: str) -> str:
        # Lógica básica para JavaScript (futuras mejoras)
        return codigo