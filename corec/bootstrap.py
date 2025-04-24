#!/usr/bin/env python3
import asyncio
import logging
import importlib
import json
from pathlib import Path
from corec.nucleus import CoreCNucleus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("corec.log")  # Guardar logs en archivo
    ]
)

async def load_plugins(nucleus: CoreCNucleus):
    """
    Carga plugins habilitados desde nucleus.config['plugins'].
    - Lee config.json de cada plugin.
    - Importa plugins.<name>.main.
    - Llama a inicializar(nucleus, config_plugin).
    """
    plugins_conf = nucleus.config.get("plugins", {})
    for name, info in plugins_conf.items():
        if not info.get("enabled", False):
            nucleus.logger.info(f"[Bootstrap] Plugin '{name}' deshabilitado, omitiendo")
            continue
        try:
            plugin_conf = {}
            conf_path = info.get("path")
            if conf_path and Path(conf_path).is_file():
                with open(conf_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    plugin_conf = raw.get(name, {})
            else:
                nucleus.logger.warning(f"[Bootstrap] Configuración de plugin '{name}' no encontrada en {conf_path}")

            module_path = f"plugins.{name}.main"
            plugin_mod = importlib.import_module(module_path)
            init_fn = getattr(plugin_mod, "inicializar", None)
            if callable(init_fn):
                await init_fn(nucleus, plugin_conf)
                nucleus.logger.info(f"[Bootstrap] Plugin '{name}' cargado correctamente")
            else:
                nucleus.logger.error(f"[Bootstrap] Plugin '{name}' no expone inicializar()")
        except ImportError as e:
            nucleus.logger.error(f"[Bootstrap] Error importando plugin '{name}': {e}")
        except json.JSONDecodeError as e:
            nucleus.logger.error(f"[Bootstrap] Error en config.json de plugin '{name}': {e}")
        except Exception as e:
            nucleus.logger.error(f"[Bootstrap] Error inesperado cargando plugin '{name}': {e}")

async def main():
    """
    Punto de entrada principal para CoreC.
    - Inicializa CoreCNucleus.
    - Carga plugins.
    - Ejecuta el ciclo principal.
    - Maneja detención graceful.
    """
    nucleus = CoreCNucleus("config/corec_config.json")
    try:
        await nucleus.inicializar()
        await load_plugins(nucleus)
        await nucleus.ejecutar()
    except KeyboardInterrupt:
        nucleus.logger.info("[Bootstrap] Interrupción recibida, deteniendo...")
        await nucleus.detener()
    except Exception as e:
        nucleus.logger.error(f"[Bootstrap] Error crítico: {e}")
        await nucleus.detener()
        raise

if __name__ == "__main__":
    asyncio.run(main())
