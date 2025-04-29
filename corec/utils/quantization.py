# corec/utils/quantization.py
from corec.config import QUANTIZATION_STEP_DEFAULT

def escalar(valor: float, paso: float = QUANTIZATION_STEP_DEFAULT) -> float:
    """
    Escala un valor al múltiplo más cercano de `paso` entre -1.0 y 1.0.

    Args:
        valor (float): Valor a cuantizar.
        paso (float): Paso de cuantización (e.g., 0.05 para valores -1.0, -0.95, ...).

    Returns:
        float: Valor cuantizado.

    Raises:
        ValueError: Si el paso es menor o igual a 0.

    Examples:
        >>> escalar(0.123, paso=0.05)
        0.1
        >>> escalar(1.5, paso=0.05)
        1.0
    """
    if paso <= 0:
        raise ValueError("El paso de cuantización debe ser mayor que 0")
    val = max(-1.0, min(1.0, valor))
    múltiplos = round(val / paso)
    return múltiplos * paso
