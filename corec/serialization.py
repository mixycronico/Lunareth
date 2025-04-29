import struct
from typing import Dict, Any


MESSAGE_FORMAT = "!Ibf?"  # ID uint32, canal uint8, valor float32, estado bool


async def serializar_mensaje(id: int, canal: int, valor: float, activo: bool) -> bytes:
    """Serializa un mensaje binario siguiendo MESSAGE_FORMAT.

    Args:
        id (int): Identificador del mensaje.
        canal (int): Canal de comunicación.
        valor (float): Valor del mensaje.
        activo (bool): Estado del mensaje.

    Returns:
        bytes: Mensaje serializado.

    Raises:
        struct.error: Si los datos no coinciden con el formato.
    """
    try:
        return struct.pack(MESSAGE_FORMAT, id, canal, valor, activo)
    except struct.error as e:
        raise struct.error(f"Error serializando mensaje: {e}")


async def deserializar_mensaje(mensaje: bytes) -> Dict[str, Any]:
    """Deserializa un mensaje binario a un diccionario.

    Args:
        mensaje (bytes): Mensaje serializado.

    Returns:
        Dict[str, Any]: Diccionario con claves id, canal, valor, activo.

    Raises:
        struct.error: Si el mensaje no coincide con el formato.
    """
    try:
        id_, canal, valor, activo = struct.unpack(MESSAGE_FORMAT, mensaje)
        return {"id": id_, "canal": canal, "valor": valor, "activo": activo}
    except struct.error as e:
        raise struct.error(f"Error deserializando mensaje: {e}")
