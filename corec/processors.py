#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corec/processors.py
Procesadores para entidades en CoreC.
"""
from corec.core import random, logging
from typing import Dict, Any

logger = logging.getLogger("Procesadores")

class ProcesadorBase:
    async def procesar(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

class ProcesadorSensor(ProcesadorBase):
    async def procesar(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        valores = datos.get("valores", [])
        valor = sum(valores) / len(valores) if valores else random.random()
        logger.debug(f"ProcesadorSensor procesó datos: valor={valor}")
        return {"valor": valor}

class ProcesadorFiltro(ProcesadorBase):
    async def procesar(self, datos: Dict[str, Any]) -> Dict[str, Any]:
        valor = datos.get("valor", random.random())
        umbral = datos.get("umbral", 0.5)
        resultado = valor if valor > umbral else 0.0
        logger.debug(f"ProcesadorFiltro procesó datos: valor={valor}, umbral={umbral}, resultado={resultado}")
        return {"valor": resultado}