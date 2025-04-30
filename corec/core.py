import logging
from typing import Any, Dict
from corec.utils.logging import CoreCLogger


class ComponenteBase:
    """Clase base para componentes de CoreC."""
    def __init__(self):
        self.logger = logging.getLogger("CoreC")

    async def inicializar(self, nucleus: Any, config: Dict[str, Any] = None):
        """Inicializa el componente."""
        self.nucleus = nucleus
        self.logger = nucleus.logger

    async def manejar_comando(self, comando: Dict[str, Any]) -> Any:
        """Maneja un comando enviado al componente."""
        pass

    async def detener(self):
        """Detiene el componente."""
        pass


class PluginCommand:
    """Clase para comandos de plugins."""
    def __init__(self, action: str, params: Dict[str, Any]):
        self.action = action
        self.params = params or {}
