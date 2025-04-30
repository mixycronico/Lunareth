import math


def escalar(valor: float, paso: float) -> float:
    """Escala un valor a un paso de cuantización.

    Args:
        valor (float): Valor a escalar.
        paso (float): Paso de cuantización (debe ser mayor que 0).

    Returns:
        float: Valor escalado.

    Raises:
        ValueError: Si el paso es menor o igual a 0.
    """
    if paso <= 0:
        raise ValueError("El paso de cuantización debe ser mayor que 0")
    if valor > 1.0:
        return 1.0
    if valor < -1.0:
        return -1.0
    return round(valor / paso) * paso
