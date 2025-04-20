from typing import Callable, Any
import random


class Entidad:
    def __init__(self, id: str, canal: int, funcion: Callable[[float], dict]):
        self.id = id
        self.canal = canal
        self.funcion = funcion
        self.estado = "activa"

    async def procesar(self, carga: float) -> dict:
        """Procesa la entidad y devuelve un resultado."""
        try:
            resultado = await self.funcion()
            return resultado
        except Exception as e:
            self.estado = "inactiva"
            return {"valor": 0, "error": str(e)}


def crear_entidad(id: str, canal: int, funcion: Callable[[float], dict]) -> Entidad:
    """Crea una entidad con el ID, canal y función dados."""
    return Entidad(id, canal, funcion)
