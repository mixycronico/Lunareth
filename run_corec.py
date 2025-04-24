# main.py
#!/usr/bin/env python3
import asyncio
import logging
import json
from pathlib import Path
from corec.nucleus import CoreCNucleus
from plugins.registry import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("corec.log")
    ]
)

async def load_plugins(nucleus: CoreCNucleus):
    """
    Escanea nucleus.config['plugins'] y para cada plugin enabled:
      1. Carga su config.json.
      2. Usa el registro para importar y inicializar el complemento.
    """
    plugins_conf = nucleus.config.get("plugins", {})
    for name, info in plugins_conf.items():
        if not info.get("enabled", False):
            nucleus.logger.info(f"[Bootstrap] Complemento '{name}' deshabilitado, omitiendo")
            continue

        try:
            plugin_conf = {}
            conf_path = info.get("path")
            if conf_path and Path(conf_path).is_file():
                with open(conf_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    plugin_conf = raw.get(name, {})
            else:
                nucleus.logger.warning(f"[Bootstrap] Configuración de complemento '{name}' no encontrada en {conf_path}")

            await registry.load_plugin(nucleus, name, plugin_conf)
            nucleus.logger.info(f"[Bootstrap] Plugin '{name}' cargado correctamente")
        except Exception as e:
            nucleus.logger.error(f"[Bootstrap] Error cargando plugin '{name}': {e}")

async def main():
    """Punto de entrada principal para CoreC."""
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
