import struct
from typing import Dict, Any


MESSAGE_FORMAT = "!Ibf?"  # ID uint32, canal uint8, valor float32, estado bool


async def serializar_mensaje(id: int, canal: int, valor: float, activo: bool) -> bytes:
    """Serializa un mensaje binario siguiendo MESSAGE_FORMAT."""
    return struct.pack(MESSAGE_FORMAT, id, canal, valor, activo)


async def deserializar_mensaje(mensaje: bytes) -> Dict[str, Any]:
    """Deserializa un mensaje binario a un dict con keys: id, canal, valor, activo."""
    id_, canal, valor, activo = struct.unpack(MESSAGE_FORMAT, mensaje)
    return {"id": id_, "canal": canal, "valor": valor, "activo": activo}
