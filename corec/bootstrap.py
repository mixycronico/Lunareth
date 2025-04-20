#!/usr/bin/env python3
import asyncio
import logging
import importlib
import json
from pathlib import Path

from corec.nucleus import CoreCNucleus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

async def load_plugins(nucleus: CoreCNucleus):
    """
    Escanea nucleus.config['plugins'] y para cada plugin enabled:
      1. Carga su config.json.
      2. Importa plugins.<name>.main.
      3. Llama a inicializar(nucleus, config_plugin).
    """
    plugins_conf = nucleus.config.get("plugins", {})
    for name, info in plugins_conf.items():
        if not info.get("enabled", False):
            continue

        try:
            # 1) Leer configuración particular del plugin
            plugin_conf = {}
            conf_path = info.get("path")
            if conf_path and Path(conf_path).is_file():
                with open(conf_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    plugin_conf = raw.get(name, {})

            # 2) Importar módulo de arranque del plugin
            module_path = f"plugins.{name}.main"
            plugin_mod = importlib.import_module(module_path)

            # 3) Ejecutar su función inicializar
            init_fn = getattr(plugin_mod, "inicializar", None)
            if callable(init_fn):
                await init_fn(nucleus, plugin_conf)
                nucleus.logger.info(f"[Bootstrap] Plugin '{name}' cargado correctamente")
            else:
                nucleus.logger.warning(f"[Bootstrap] Plugin '{name}' no expone inicializar()")

        except Exception as e:
            nucleus.logger.error(f"[Bootstrap] Error cargando plugin '{name}': {e}")


async def main():
    # 1) Arrancar núcleo
    nucleus = CoreCNucleus("config/corec_config.json")
    try:
        await nucleus.inicializar()

        # 2) Cargar todos los plugins habilitados
        await load_plugins(nucleus)

        # 3) Entrar en bucle de ejecución (módulos + plugins)
        await nucleus.ejecutar()

    except KeyboardInterrupt:
        await nucleus.detener()


if __name__ == "__main__":
    asyncio.run(main())