import importlib
import json
from pathlib import Path
from corec.nucleus import CoreCNucleus


class PluginRegistry:
    async def load_plugin(self, nucleus: CoreCNucleus, name: str, config: dict):
        """Carga un plugin desde plugins.<name>.main.

        Args:
            nucleus (CoreCNucleus): Instancia del núcleo de CoreC.
            name (str): Nombre del plugin.
            config (dict): Configuración del plugin.

        Raises:
            ImportError: Si el módulo del plugin no puede importarse.
            AttributeError: Si el plugin no tiene una función inicializar.
            ValueError: Si la configuración del plugin es inválida.
        """
        try:
            module_path = f"plugins.{name}.main"
            plugin_mod = importlib.import_module(module_path)
            init_fn = getattr(plugin_mod, "inicializar", None)
            if not callable(init_fn):
                raise AttributeError(f"Plugin '{name}' no expone función inicializar")
            await init_fn(nucleus, config)
            nucleus.logger.info(f"Plugin '{name}' cargado correctamente")
        except ImportError as e:
            nucleus.logger.error(f"Error importando plugin '{name}': {e}")
            raise
        except AttributeError as e:
            nucleus.logger.error(f"Error en plugin '{name}': {e}")
            raise
        except Exception as e:
            nucleus.logger.error(f"Error inesperado cargando plugin '{name}': {e}")
            raise
