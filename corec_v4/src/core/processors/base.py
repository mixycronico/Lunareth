from abc import ABC, abstractmethod
from typing import Dict, Any

class ProcesadorBase(ABC):
    def __init__(self):
        self.nucleus = None

    def set_nucleus(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus

    @abstractmethod
    async def procesar(self, datos: Any, contexto: Dict[str, Any]) -> Any:
        pass