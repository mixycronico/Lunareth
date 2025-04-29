import asyncio
import json
import logging
from pathlib import Path
import psutil
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from corec.nucleus import CoreCNucleus
from corec.utils.db_utils import init_postgresql, init_redis
from corec.utils.logging import setup_logging
from plugins.registry import registry


async def log_system_metrics(logger: logging.Logger):
    """Registra métricas básicas del sistema periódicamente.

    Args:
        logger (logging.Logger): Logger para registrar las métricas.
    """
    while True:
        try:
            cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            mem_percent = mem.used / mem.total * 100
            logger.info(f"CPU: {cpu_percent:.1f}%, Memoria: {mem_percent:.1f}% ({mem.used / 1024**2:.1f} MB / {mem.total / 1024**2:.1f} MB)")
            await asyncio.sleep(300)  # Cada 5 minutos
        except Exception as e:
            logger.error(f"Error registrando métricas: {e}")
            await asyncio.sleep(60)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.getLogger("CoreCBootstrap").info(
        f"Reintentando carga de plugin... Intento {retry_state.attempt_number}"
    )
)
async def load_plugins(nucleus: CoreCNucleus):
    """Carga plugins habilitados desde la configuración.

    Args:
        nucleus (CoreCNucleus): Instancia del núcleo de CoreC.

    Raises:
        ValueError: Si la configuración del plugin es inválida.
    """
    plugins_conf = nucleus.config.plugins
    if not plugins_conf:
        nucleus.logger.warning("No se encontraron plugins en la configuración")
        return

    for name, info in plugins_conf.items():
        if not info.enabled:
            nucleus.logger.info(f"Plugin '{name}' deshabilitado, omitiendo")
            continue

        try:
            plugin_conf = info.model_dump()
            conf_path = info.path
            if conf_path and Path(conf_path).is_file():
                with open(conf_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                    if name not in raw:
                        raise ValueError(f"Plugin '{name}' configuration missing in {conf_path}")
                    plugin_conf.update(raw[name])
            else:
                nucleus.logger.warning(f"Configuración de plugin '{name}' no encontrada en {conf_path}, usando configuración base")

            await registry.load_plugin(nucleus, name, plugin_conf)
            nucleus.logger.info(f"Plugin '{name}' cargado correctamente")
        except Exception as e:
            nucleus.logger.error(f"Error cargando plugin '{name}': {e}")
            raise


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logging.getLogger("CoreCBootstrap").info(
        f"Reintentando inicialización del núcleo... Intento {retry_state.attempt_number}"
    )
)
async def initialize_nucleus(config_path: str) -> CoreCNucleus:
    """Inicializa el núcleo de CoreC con reintentos.

    Args:
        config_path (str): Ruta al archivo de configuración.

    Returns:
        CoreCNucleus: Instancia inicializada del núcleo.

    Raises:
        FileNotFoundError: Si el archivo de configuración no existe.
    """
    nucleus = CoreCNucleus(config_path)
    nucleus.db_pool = await init_postgresql(nucleus.config.db_config.model_dump())
    nucleus.redis_client = await init_redis(nucleus.config.redis_config.model_dump())
    await nucleus.inicializar()
    return nucleus


async def main():
    """Punto de entrada principal para CoreC."""
    config_path = "config/corec_config.json"
    try:
        from corec.config_loader import load_config
        config = load_config(config_path).model_dump()
        logger = setup_logging(config)
        nucleus = await initialize_nucleus(config_path)
        asyncio.create_task(log_system_metrics(logger))
        await load_plugins(nucleus)
        await nucleus.ejecutar()
    except KeyboardInterrupt:
        logger.info("Interrupción recibida, deteniendo...")
        await nucleus.detener()
    except FileNotFoundError as e:
        logger.error(f"Archivo de configuración no encontrado: {e}")
        raise
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        await nucleus.detener()
        raise


if __name__ == "__main__":
    asyncio.run(main())
