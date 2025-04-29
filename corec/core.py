from typing import Any, Dict


class ComponenteBase:
    """Clase base para componentes de CoreC."""

    async def inicializar(self, nucleus: Any, config: Dict[str, Any] = None):
        """Inicializa el componente.

        Args:
            nucleus: Instancia del núcleo de CoreC.
            config: Configuración del componente (opcional).
        """
        pass

    async def manejar_comando(self, comando: Dict[str, Any]) -> Any:
        """Maneja un comando enviado al componente.

        Args:
            comando: Diccionario con la acción y parámetros del comando.

        Returns:
            Any: Resultado del procesamiento del comando.
        """
        pass

    async def detener(self):
        """Detiene el componente."""
        pass


class PluginCommand:
    """Clase para comandos de plugins."""
    def __init__(self, action: str, params: Dict[str, Any]):
        self.action = action
        self.params = params or {}
