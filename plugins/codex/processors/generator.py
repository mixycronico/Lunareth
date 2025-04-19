#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/processors/generator.py
Genera websites y plugins en el plugin Codex.
"""
import os
import shutil
import logging
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any

class CodexGenerator:
    def __init__(self, config):
        self.logger = logging.getLogger("CodexGenerator")
        self.config = config
        self.website_output_dir = config.get("website_output_dir", "generated_websites/")
        self.plugin_output_dir = config.get("plugin_output_dir", "plugins/")
        self.website_templates = config.get("website_templates", {})
        self.plugin_templates = config.get("plugin_templates", {})
        self.env = Environment(loader=FileSystemLoader("plugins/codex/utils/templates"))

    async def generar_website(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            template_type = params.get("template", "react")
            project_name = params.get("project_name", "website")
            output_dir = os.path.join(self.website_output_dir, project_name)
            template_dir = self.website_templates.get(template_type, "utils/templates/react_app")

            shutil.copytree(template_dir, output_dir, dirs_exist_ok=True)
            self._render_templates(output_dir, params)
            self.logger.info(f"Website generado: {output_dir}")
            return {"status": "ok", "output_dir": output_dir}
        except Exception as e:
            self.logger.error(f"Error generando website: {e}")
            return {"status": "error", "message": str(e)}

    async def generar_plugin(self, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            plugin_name = params.get("plugin_name", "new_plugin")
            template_type = params.get("template", "corec_plugin")
            output_dir = os.path.join(self.plugin_output_dir, plugin_name)
            template_dir = self.plugin_templates.get(template_type, "utils/templates/plugin")

            shutil.copytree(template_dir, output_dir, dirs_exist_ok=True)
            self._render_templates(output_dir, params)
            self.logger.info(f"Plugin generado: {output_dir}")
            return {"status": "ok", "output_dir": output_dir}
        except Exception as e:
            self.logger.error(f"Error generando plugin: {e}")
            return {"status": "error", "message": str(e)}

    def _render_templates(self, output_dir: str, params: Dict[str, Any]):
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith((".py", ".js", ".html", ".css", ".json")):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, output_dir)
                    template = self.env.get_template(rel_path)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(template.render(**params))