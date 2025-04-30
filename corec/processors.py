import logging
from typing import Dict, Any


logger = logging.getLogger("corec.processors")


class ProcesadorBase:
    async def procesar(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa datos de entrada."""
        raise NotImplementedError


class ProcesadorSensor(ProcesadorBase):
    async def procesar(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa datos de sensores, promediando valores o generando un valor aleatorio."""
        vals = datos.get("valores", [])
        v = sum(vals) / len(vals) if vals else random.random()
        logger.debug(f"Sensor procesado: {v:.4f}")
        return {"valor": v}


class ProcesadorFiltro(ProcesadorBase):
    async def procesar(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        """Filtra valores basándose en un umbral."""
        v = datos.get("valor", random.random())
        um = datos.get("umbral", 0.5)
        out = v if v > um else 0.0
        logger.debug(f"Filtro aplicado: {v:.4f} vs {um:.2f} -> {out:.4f}")
        return {"valor": out}
