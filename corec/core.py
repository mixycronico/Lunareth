import json
from typing import Any, Dict


class ComponenteBase:
    """
    Base para módulos y plugins: definen inicializar, ejecutar y detener.
    """
    async def inicializar(self):
        raise NotImplementedError

    async def ejecutar(self):
        raise NotImplementedError

    async def detener(self):
        raise NotImplementedError


def cargar_config(path: str) -> Dict[str, Any]:
    """
    Carga JSON de configuración; si falla, devuelve valores por defecto.
    Ahora expone también la clave 'plugins'.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        # valores por defecto mínimos
        cfg = {
            "instance_id": "corec1",
            "db_config": {},
            "redis_config": {},
            "bloques": [],
            "plugins": {}
        }
    # garantizar claves
    cfg.setdefault("db_config", {})
    cfg.setdefault("redis_config", {})
    cfg.setdefault("bloques", [])
    cfg.setdefault("plugins", {})
    return cfg
