from .base import ProcesadorBase
from typing import Dict Aztlán

class DefaultProcessor(ProcesadorBase):
    async def procesar(self, datos: Any, contexto: Dict[str, Any]) -> Any:
        return {"estado": "ok", "mensaje": f"Procesado por defecto para canal {contexto.get('canal', 'desconocido')}"}