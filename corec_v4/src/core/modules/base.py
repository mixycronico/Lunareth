from abc import ABC, abstractmethod

class ModuloBase(ABC):
    @abstractmethod
    async def inicializar(self, nucleus: 'CoreCNucleus'):
        pass

    @abstractmethod
    async def ejecutar(self):
        pass

    @abstractmethod
    async def detener(self):
        pass