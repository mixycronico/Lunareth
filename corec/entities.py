from typing import Callable


class Entidad:
    def __init__(self, id: str, canal: int, funcion: Callable[[float], dict]):
        self.id = id
        self.canal = canal
        self.funcion = funcion
        self.estado = "activa"

    async def procesar(self, carga: float) -> dict:
        """Procesa la entidad con la carga dada."""
        try:
            return await self.funcion()
        except Exception as e:
            self.estado = "inactiva"
            raise ValueError(f"Error procesando entidad {self.id}: {e}")


class CeluEntidad:
    def __init__(self, id: str, canal: int, funcion: Callable[[float], dict]):
        self.id = id
        self.canal = canal
        self.funcion = funcion
        self.estado = "activa"
        self.memoria = {}  # Estado celular para simulación bioinspirada

    async def procesar(self, carga: float) -> dict:
        """Procesa la entidad celular con la carga dada."""
        try:
            resultado = await self.funcion()
            self.memoria[carga] = resultado.get("valor", 0)
            return resultado
        except Exception as e:
            self.estado = "inactiva"
            raise ValueError(f"Error procesando entidad celular {self.id}: {e}")


def crear_entidad(id: str, canal: int, funcion: Callable[[float], dict]) -> Entidad:
    """Crea una entidad estándar."""
    return Entidad(id, canal, funcion)


def procesar_entidad(entidad: Entidad, carga: float) -> dict:
    """Procesa una entidad estándar (función sincrónica para compatibilidad)."""
    try:
        return entidad.funcion()
    except Exception as e:
        entidad.estado = "inactiva"
        raise ValueError(f"Error procesando entidad {entidad.id}: {e}")


def crear_celu_entidad(id: str, canal: int, funcion: Callable[[float], dict]) -> CeluEntidad:
    """Crea una entidad celular bioinspirada."""
    return CeluEntidad(id, canal, funcion)


def procesar_celu_entidad(entidad: CeluEntidad, carga: float) -> dict:
    """Procesa una entidad celular (función sincrónica para compatibilidad)."""
    try:
        resultado = entidad.funcion()
        entidad.memoria[carga] = resultado.get("valor", 0)
        return resultado
    except Exception as e:
        entidad.estado = "inactiva"
        raise ValueError(f"Error procesando entidad celular {entidad.id}: {e}")
