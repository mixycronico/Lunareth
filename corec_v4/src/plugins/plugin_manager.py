import importlib
import json
from pathlib import Path
from typing import Dict, Any
from ..utils.logging import logger
from ..utils.config import load_secrets
from ..core.processors.base import ProcesadorBase

class PluginBase:
    def __init__(self, config: Dict[str, Any], redis_client=None, db_config: Dict[str, Any] = None):
        self.config = config
        self.redis_client = redis_client
        self.db_config = db_config
        self.logger = logger.getLogger(f"Plugin-{self.config.get('name', 'unknown')}")

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        pass

    async def detener(self):
        pass

class PluginManager:
    def __init__(self, plugins_dir: str = "src/plugins", config_dir: str = "configs/plugins"):
        self.plugins_dir = Path(plugins_dir)
        self.config_dir = Path(config_dir)
        self.plugins: Dict[str, PluginBase] = {}
        self.logger = logger.getLogger("PluginManager")
        self.redis_client = None
        self.db_config = None

    async def cargar_plugins(self, nucleus: 'CoreCNucleus'):
        self.redis_client = nucleus.redis_client
        self.db_config = nucleus.db_config
        for plugin_dir in self.plugins_dir.iterdir():
            if plugin_dir.is_dir() and (plugin_dir / "plugin.json").exists():
                try:
                    with open(plugin_dir / "plugin.json", "r") as f:
                        config = json.load(f)
                    plugin_type = config.get("type", "processor")
                    module_path = config["main_class"].split(".")
                    module = importlib.import_module(".".join(module_path[:-1]))
                    plugin_class = getattr(module, module_path[-1])
                    plugin_config = load_secrets(self.config_dir / config["config_file"])
                    plugin = plugin_class(
                        config=plugin_config,
                        redis_client=self.redis_client,
                        db_config=self.db_config
                    )
                    await plugin.inicializar(nucleus)
                    self.plugins[config["name"]] = plugin
                    self.register_plugin_channels(nucleus, config)
                    self.logger.info(f"Plugin {config['name']} ({plugin_type}) cargado")
                except Exception as e:
                    self.logger.error(f"Error cargando plugin {plugin_dir.name}: {e}")

    def register_plugin_channels(self, nucleus: 'CoreCNucleus', config: Dict[str, Any]):
        for channel in config.get("channels", []):
            if config.get("critical", False):
                nucleus.canales_criticos.append(channel)

    async def detener_plugins(self):
        for plugin_name, plugin in self.plugins.items():
            await plugin.detener()
            self.logger.info(f"Plugin {plugin_name} detenido")

    def get_processor(self, channel: str) -> ProcesadorBase:
        for plugin in self.plugins.values():
            if isinstance(plugin, ProcesadorBase) and channel in plugin.config.get("channels", []):
                return plugin
        return None

    def get_interface(self, interface_name: str) -> Any:
        for plugin in self.plugins.values():
            if plugin.config.get("type") == "interface" and plugin.config.get("name") == interface_name:
                return plugin
        return None