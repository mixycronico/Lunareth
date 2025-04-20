import logging
from corec.core import ComponenteBase


async def inicializar(nucleus, config):
    """Inicializa el plugin genérico."""
    plugin = ExamplePlugin()
    await plugin.inicializar(nucleus, config)
    return plugin


class ExamplePlugin(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ExamplePlugin")
        self.nucleus = None
        self.config = None

    async def inicializar(self, nucleus, config):
        """Inicializa el plugin."""
        self.nucleus = nucleus
        self.config = config
        self.nucleus.registrar_plugin("example_plugin", self)
        self.logger.info("[ExamplePlugin] Inicializado")

    async def manejar_comando(self, comando):
        """Maneja un comando recibido."""
        try:
            return {"status": "success", "action": comando["action"]}
        except Exception as e:
            self.logger.error(f"[ExamplePlugin] Error manejando comando: {e}")
            return {"status": "error", "message": str(e)}

    async def detener(self):
        """Detiene el plugin."""
        self.logger.info("[ExamplePlugin] Detenido")
