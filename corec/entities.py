# corec/entities.py
from typing import Callable, Dict, Any, Tuple
from corec.serialization import serializar_mensaje, deserializar_mensaje

MicroCeluEntidadCoreC = Tuple[str, int, Callable[[], Any], bool]
CeluEntidadCoreC       = Tuple[str, int, Callable[[Dict[str, Any]], Any], bool]

def crear_entidad(id: str, canal: int, funcion: Callable[[], Any]) -> MicroCeluEntidadCoreC:
    """Crea una entidad micro que ejecuta una función y retorna bytes serializados."""
    return (id, canal, funcion, True)

async def procesar_entidad(entidad: MicroCeluEntidadCoreC, umbral: float = 0.5) -> bytes:
    """Procesa una entidad micro y serializa su resultado con threshold umbral."""
    id_, canal, funcion, activo = entidad
    if not activo:
        return await serializar_mensaje(int(id_[1:]) if id_.startswith("m") else int(id_), canal, 0.0, False)
    try:
        resultado = await funcion()
        valor    = float(resultado.get("valor", 0.0))
        ok       = valor > umbral
        return await serializar_mensaje(int(id_[1:]) if id_.startswith("m") else int(id_), canal, valor if ok else 0.0, ok)
    except Exception:
        return await serializar_mensaje(int(id_[1:]) if id_.startswith("m") else int(id_), canal, 0.0, False)

def crear_celu_entidad(id: str, canal: int, procesador: Callable[[Dict[str, Any]], Any]) -> CeluEntidadCoreC:
    """Crea una entidad celular que procesa datos y retorna bytes serializados."""
    return (id, canal, procesador, True)

async def procesar_celu_entidad(entidad: CeluEntidadCoreC, datos: Dict[str, Any], umbral: float = 0.5) -> bytes:
    """Procesa una entidad celular y serializa su resultado con threshold umbral."""
    id_, canal, procesador, activo = entidad
    if not activo:
        return await serializar_mensaje(int(id_[1:]) if id_.startswith("c") else int(id_), canal, 0.0, False)
    try:
        resultado = await procesador(datos)
        valor    = float(resultado.get("valor", 0.0))
        ok       = valor > umbral
        return await serializar_mensaje(int(id_[1:]) if id_.startswith("c") else int(id_), canal, valor if ok else 0.0, ok)
    except Exception:
        return await serializar_mensaje(int(id_[1:]) if id_.startswith("c") else int(id_), canal, 0.0, False)