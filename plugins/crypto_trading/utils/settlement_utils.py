# plugins/crypto_trading/utils/settlement_utils.py

def calcular_ganancias(op: dict) -> dict:
    """Calcula ganancia o pérdida porcentual."""
    entrada = float(op["precio_entrada"])
    salida  = float(op["precio_salida"])
    qty     = float(op["cantidad"])
    ganancia_pct = ((salida - entrada) / entrada) * 100
    ganancia_total = (salida - entrada) * qty
    return {
        "symbol": op["symbol"],
        "ganancia_pct": round(ganancia_pct, 2),
        "ganancia_total": round(ganancia_total, 2),
        "cantidad": qty,
        "entrada": entrada,
        "salida": salida,
        "usuario_id": op.get("usuario_id", "pool"),
        "timestamp": op.get("timestamp"),
    }

async def registrar_historial(redis, key: str, resultado: dict):
    """Guarda historial de operación finalizada."""
    await redis.rpush(key, resultado)
