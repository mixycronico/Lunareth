from typing import Callable, Dict, Any


class Entidad:
    def __init__(self, id: str, canal: int, procesar: Callable[[float], Dict[str, Any]]):
        self.id = id
        self.canal = canal
        self._procesar = procesar
        self.estado = "activa"  # Agregamos el atributo estado

    async def procesar(self, carga: float) -> Dict[str, Any]:
        """Procesa la entidad con una carga dada."""
        return self._procesar(carga)


def crear_entidad(id: str, canal: int, procesar: Callable[[float], Dict[str, Any]]) -> Entidad:
    """Crea una entidad con la función de procesamiento dada."""
    return Entidad(id, canal, procesar)
