from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
from corec.utils.quantization import escalar


class EntidadBase(ABC):
    """Interfaz para entidades procesables en CoreC."""
    @abstractmethod
    async def procesar(self, carga: float) -> Dict[str, Any]:
        pass

    @abstractmethod
    def recibir_cambio(self, cambio: Dict[str, float]):
        pass


class Entidad(EntidadBase):
    def __init__(
        self,
        id: str,
        canal: int,
        procesar: Callable[[float], Dict[str, Any]],
        quantization_step: float = 0.05
    ):
        """Entidad básica que procesa datos y recibe cambios cuantizados.

        Args:
            id (str): Identificador único.
            canal (int): Canal de comunicación.
            procesar (Callable): Función de procesamiento.
            quantization_step (float): Paso de cuantización específico.
        """
        self.id = id
        self.canal = canal
        self._procesar = procesar
        self.estado = "activa"
        self.caracteristicas: Dict[str, float] = {}
        self.quantization_step = quantization_step

    async def procesar(self, carga: float) -> Dict[str, Any]:
        """Procesa la entidad y cuantiza el valor resultante.

        Args:
            carga (float): Factor de carga (0.0 a 1.0).

        Returns:
            Dict[str, Any]: Resultado procesado con valor cuantizado.
        """
        resultado = self._procesar(carga)
        if "valor" in resultado and isinstance(resultado["valor"], (int, float)):
            resultado["valor"] = escalar(resultado["valor"], self.quantization_step)
        return resultado

    def recibir_cambio(self, cambio: Dict[str, float]):
        """Actualiza características con valores cuantizados.

        Args:
            cambio (Dict[str, float]): Cambios a aplicar.
        """
        cambio_escalado = {k: escalar(v, self.quantization_step) for k, v in cambio.items()}
        self.caracteristicas.update(cambio_escalado)
