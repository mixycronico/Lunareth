import logging
import uuid
from typing import Optional
from pathlib import Path


class CoreCLogger(logging.Logger):
    """Logger personalizado para CoreC con soporte para IDs de correlación."""

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, correlation_id: Optional[str] = None):
        """Añade un ID de correlación al mensaje de log.

        Args:
            level: Nivel de log.
            msg: Mensaje a registrar.
            args: Argumentos para el mensaje.
            exc_info: Información de la excepción (opcional).
            extra: Datos adicionales (opcional).
            stack_info: Información de la pila (opcional).
            correlation_id: ID de correlación para rastrear eventos (opcional).
        """
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]
        msg = f"[CorrelationID: {correlation_id}] {msg}"
        super()._log(level, msg, args, exc_info, extra, stack_info)


def setup_logging(config: dict = None) -> logging.Logger:
    """Configura el logging para CoreC con manejadores de consola y archivo.

    Args:
        config (dict, optional): Configuración con log_level y log_file. Usa valores por defecto si no se proporciona.

    Returns:
        logging.Logger: Logger raíz configurado para CoreC.
    """
    # Establecer la clase personalizada para todos los loggers
    logging.setLoggerClass(CoreCLogger)

    # Configuración por defecto
    log_level = config.get("log_level", "INFO").upper() if config else "INFO"
    log_file = config.get("log_file", "corec.log") if config else "corec.log"

    # Validar nivel de log
    log_level = getattr(logging, log_level, logging.INFO)

    # Configurar el logger raíz
    logger = logging.getLogger("CoreC")
    logger.setLevel(log_level)

    # Evitar duplicación de manejadores
    logger.handlers.clear()

    # Formato de log
    log_format = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    # Manejador de consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # Manejador de archivo
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    logger.info(f"Logging configurado con nivel {log_level}, archivo: {log_file}")
    return logger
