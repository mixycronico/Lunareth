# plugins/registry.py
from typing import Dict, Any
import logging
import importlib

logger = logging.getLogger("PluginRegistry")

class PluginRegistry:
    def __init__(self):
        self.plugins: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, module_path: str, version: str):
        """Registra un complemento con su módulo y versión."""
        self.plugins[name] = {
            "module_path": module_path,
            "version": version,
            "loaded": False
        }
        logger.info(f"[Registry] Complemento registrado: {name} (v{version})")

    async def load_plugin(self, nucleus: 'CoreCNucleus', name: str, config: Dict[str, Any]):
        """Carga un complemento registrado."""
        if name not in self.plugins:
            raise ValueError(f"Complemento {name} no está registrado")
        plugin_info = self.plugins[name]
        try:
            plugin_mod = importlib.import_module(plugin_info["module_path"])
            init_fn = getattr(plugin_mod, "inicializar", None)
            if not callable(init_fn):
                raise ValueError(f"Complemento {name} no expone inicializar()")
            await init_fn(nucleus, config)
            plugin_info["loaded"] = True
            logger.info(f"[Registry] Complemento {name} cargado correctamente")
        except Exception as e:
            logger.error(f"[Registry] Error cargando complemento {name}: {e}")
            raise

registry = PluginRegistry()

registry.register("codex", "plugins.codex.main", "1.0.0")
registry.register("comm_system", "plugins.comm_system.main", "1.0.0")
registry.register("crypto_trading", "plugins.crypto_trading.main", "1.0.0")
registry.register("panel_control", "plugins.panel_control.main", "1.0.0")
