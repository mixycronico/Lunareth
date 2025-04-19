from corec.core import random, asyncio, struct, serializar_mensaje, deserializar_mensaje
from typing import Callable, Dict, Any

MicroCeluEntidadCoreC = tuple[str, int, Callable[[], Any], bool]

def crear_entidad(id: str, canal: int, funcion: Callable[[], Any]) -> MicroCeluEntidadCoreC:
    return (id, canal, funcion, True)

async def procesar_entidad(entidad: MicroCeluEntidadCoreC, umbral: float = 0.5) -> bytes:
    id, canal, funcion, activo = entidad
    if not activo:
        return await serializar_mensaje(int(id[1:]), canal, 0.0, False)
    try:
        resultado = await funcion()
        valor = resultado["valor"]
        if valor > umbral:
            return await serializar_mensaje(int(id[1:]), canal, valor, True)
        return await serializar_mensaje(int(id[1:]), canal, 0.0, True)
    except Exception as e:
        return await serializar_mensaje(int(id[1:]), canal, 0.0, False)

CeluEntidadCoreC = tuple[str, int, Callable[[Dict[str, Any]], Any], bool]

def crear_celu_entidad(id: str, canal: int, procesador: Callable[[Dict[str, Any]], Any]) -> CeluEntidadCoreC:
    return (id, canal, procesador, True)

async def procesar_celu_entidad(entidad: CeluEntidadCoreC, datos: Dict[str, Any], umbral: float = 0.5) -> bytes:
    id, canal, procesador, activo = entidad
    if not activo:
        return await serializar_mensaje(int(id[1:]), canal, 0.0, False)
    try:
        resultado = await procesador(datos)
        valor = resultado["valor"]
        if valor > umbral:
            return await serializar_mensaje(int(id[1:]), canal, valor, True)
        return await serializar_mensaje(int(id[1:]), canal, 0.0, True)
    except Exception as e:
        return await serializar_mensaje(int(id[1:]), canal, 0.0, False)