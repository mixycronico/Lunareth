#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/codex/processors/generator.py
Genera plugins y websites desde plantillas Jinja2.
"""
import os, shutil, logging
from jinja2 import Environment, FileSystemLoader
from typing import Dict, Any
from utils.helpers import run_blocking

class Generator:
    def __init__(self, config: Dict[str,Any]):
        self.logger     = logging.getLogger("CodexGenerator")
        tpl_dir          = config["templates_dir"]
        self.plugin_tpl = os.path.join(tpl_dir, "plugin")
        self.web_tpls   = {
            "react": os.path.join(tpl_dir, "react_app"),
            "fastapi": os.path.join(tpl_dir, "fastapi_app")
        }
        self.out_plugins  = config["output_plugins"]
        self.out_websites = config["output_websites"]
        self.env          = Environment(loader=FileSystemLoader(tpl_dir))

    async def generate_plugin(self, params: Dict[str,Any]) -> Dict[str,Any]:
        name = params.get("plugin_name")
        dest = os.path.join(self.out_plugins, name)
        try:
            await run_blocking(shutil.copytree, self.plugin_tpl, dest, dirs_exist_ok=True)
            self._render(dest, params)
            return {"status":"ok","type":"plugin","path":dest}
        except Exception as e:
            self.logger.error(f"Error generando plugin: {e}")
            return {"status":"error","message":str(e)}

    async def generate_website(self, params: Dict[str,Any]) -> Dict[str,Any]:
        kind = params.get("template", "react")
        src  = self.web_tpls.get(kind)
        name = params.get("project_name", "webapp")
        dest = os.path.join(self.out_websites, name)
        try:
            await run_blocking(shutil.copytree, src, dest, dirs_exist_ok=True)
            self._render(dest, params)
            return {"status":"ok","type":"website","path":dest}
        except Exception as e:
            self.logger.error(f"Error generando website: {e}")
            return {"status":"error","message":str(e)}

    def _render(self, root: str, params: Dict[str,Any]):
        for base, _, files in os.walk(root):
            for fn in files:
                if fn.endswith((".py",".js",".html",".json")):
                    rel = os.path.relpath(os.path.join(base,fn), root)
                    tpl = self.env.get_template(rel)
                    content = tpl.render(**params)
                    with open(os.path.join(base,fn),"w",encoding="utf-8") as f:
                        f.write(content)