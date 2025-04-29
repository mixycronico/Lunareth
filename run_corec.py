#!/usr/bin/env python3
import asyncio
import logging
import json
from pathlib import Path
import psutil
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from corec.nucleus import CoreCNucleus
from plugins.registry import registry

def setup_logging(config: dict):
    """Configura el logging basado en la configuración."""
    log_level = config.get("log_level", "INFO").upper()
    log_file = config.get("log_file", "corec.log")
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    logger = logging.getLogger("CoreCBootstrap")
    logger.info(f"[Bootstrap] Logging configurado con nivel {log_level}, archivo: {log_file}")
    return logger

async def log_system_metrics(logger):
    """Registra métricas básicas del sistema periódicamente."""
    while True:
        try:
            cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            mem_percent = mem.used / mem.total * 100
            logger.info(f"[Metrics] CPU: {cpu_percent:.1f}%, Memoria: {mem_percent:.1f}% ({mem.used / 1024**2:.1f} MB / {mem.total / 1024**2:.1f} MB)")
            await asyncio.sleep(300)  # Cada 5 minutos
        except Exception as e:
            logger.error(f"[Metrics] Error registrando métricas: {e}")
            await asyncio.sleep(60)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.getLogger("CoreCBootstrap").info(f"Reintentando carga de plugin... Intento {retry_state.attempt_number}")
)
async def load_plugins(nucleus: CoreCNucleus):
    """
    Escanea nucleus.config['plugins'] y para cada plugin enabled:
      1. Carga su config.json.
      2. Usa el registro para importar y inicializar el complemento.
    """
    plugins_conf = nucleus.config.get("plugins", {})
    if not plugins_conf:
        nucleus.logger.warning("[Bootstrap] No se encontraron plugins en la configuración")
        return

    for name, info in plugins_conf.items():
        if not info.get("enabled", False):
            nucleus.logger.info(f"[Bootstrap] Complemento '{name}' deshabilitado, omitiendo")
            continue

        try:
            plugin_conf = info.copy()  # Copia la configuración base
            conf_path = info.get("path")
            if conf_path and Path(conf_path).is_file():
                with open(conf_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    plugin_conf.update(raw.get(name, {}))
            else:
                nucleus.logger.warning(f"[Bootstrap] Configuración de complemento '{name}' no encontrada en {conf_path}, usando configuración base")

            # Validar configuración del plugin
            required_fields = ["bloque"]
            for field in required_fields:
                if field not in plugin_conf:
                    raise ValueError(f"Configuración de plugin '{name}' incompleta: falta '{field}'")

            await registry.load_plugin(nucleus, name, plugin_conf)
            nucleus.logger.info(f"[Bootstrap] Plugin '{name}' cargado correctamente")
        except Exception as e:
            nucleus.logger.error(f"[Bootstrap] Error cargando plugin '{name}': {e}")
            raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.getLogger("CoreCBootstrap").info(f"Reintentando inicialización del núcleo... Intento {retry_state.attempt_number}")
)
async def initialize_nucleus(config_path: str) -> CoreCNucleus:
    """Inicializa el núcleo de CoreC con reintentos."""
    nucleus = CoreCNucleus(config_path)
    await nucleus.inicializar()
    return nucleus

async def main():
    """Punto de entrada principal para CoreC."""
    config_path = "config/corec_config.json"
    try:
        # Cargar configuración para logging
        config = {}
        if Path(config_path).is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            raise FileNotFoundError(f"Archivo de configuración {config_path} no encontrado")

        # Configurar logging
        logger = setup_logging(config)

        # Inicializar núcleo con reintentos
        nucleus = await initialize_nucleus(config_path)

        # Iniciar tarea de monitoreo de métricas
        asyncio.create_task(log_system_metrics(logger))

        # Cargar plugins
        await load_plugins(nucleus)

        # Ejecutar núcleo
        await nucleus.ejecutar()
    except KeyboardInterrupt:
        logger.info("[Bootstrap] Interrupción recibida, deteniendo...")
        await nucleus.detener()
    except Exception as e:
        logger.error(f"[Bootstrap] Error crítico: {e}")
        await nucleus.detener()
        raise

if __name__ == "__main__":
    asyncio.run(main())
