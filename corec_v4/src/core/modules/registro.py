#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/core/modules/registro.py
"""
registro.py
Registra y gestiona entidades CeluEntidadCoreC y MicroCeluEntidadCoreC, con soporte para millones de micro-celus.
Optimiza enjambres con particionamiento y caché en Redis.
"""

import asyncio
import time
import random
from typing import Dict, List
from collections import defaultdict
from ..utils.logging import logger
from .base import ModuloBase
from ..celu_entidad import CeluEntidadCoreC
from ..micro_celu import MicroCeluEntidadCoreC
from ..micro_nano_dna import MicroNanoDNA
import psycopg2
import redis.asyncio as aioredis
import json
import zstandard as zstd

class ModuloRegistro(ModuloBase):
    def __init__(self):
        self.logger = logger.getLogger("ModuloRegistro")
        self.celu_entidades: Dict[str, CeluEntidadCoreC] = {}
        self.micro_celu_entidades: Dict[str, MicroCeluEntidadCoreC] = {}
        self.enjambres: Dict[str, List[MicroCeluEntidadCoreC]] = defaultdict(list)
        self.sub_enjambres: Dict[str, List[List[MicroCeluEntidadCoreC]]] = defaultdict(list)
        self.max_enjambre_por_canal = 10000000  # Soporte para 10M micro-celus
        self.min_enjambre_por_canal = 100
        self.max_sub_enjambre_size = 100000  # Tamaño máximo por sub-enjambre
        self.load_threshold = 0.5  # Umbral alto
        self.low_load_threshold = 0.2  # Umbral bajo
        self.redis = None

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.redis = aioredis.from_url("redis://redis:6379")
        asyncio.create_task(self.optimize_swarms())
        self.logger.info("[ModuloRegistro] Inicializado")

    async def optimize_swarms(self):
        while True:
            try:
                conn = psycopg2.connect(**self.nucleus.db_config)
                cur = conn.cursor()
                cur.execute("SELECT AVG(carga), COUNT(*) FROM nodos WHERE instance_id = %s", (self.nucleus.instance_id,))
                load, active_nodes = cur.fetchone()
                load = load or 0
                active_nodes = active_nodes or 1
                cur.close()
                conn.close()

                # Ajustar umbrales dinámicamente según nodos activos
                dynamic_load_threshold = self.load_threshold * (1 + 0.1 * (5 - active_nodes))
                dynamic_low_threshold = self.low_load_threshold * (1 + 0.1 * (5 - active_nodes))

                for canal, enjambre in self.enjambres.items():
                    current_size = len(enjambre)
                    # Particionar enjambres grandes en sub-enjambres
                    if current_size > self.max_sub_enjambre_size:
                        await self.partition_swarm(canal)

                    # Optimizar sub-enjambres
                    for sub_enjambre in self.sub_enjambres[canal]:
                        sub_size = len(sub_enjambre)
                        if load > dynamic_load_threshold and sub_size > self.min_enjambre_por_canal:
                            await self._limpiar_sub_enjambre(canal, sub_enjambre)
                            self.logger.info(f"Sub-enjambre {canal} reducido: {sub_size} -> {len(sub_enjambre)}")
                        elif load < dynamic_low_threshold and sub_size < self.max_sub_enjambre_size:
                            await self.regenerar_sub_enjambre(canal, sub_enjambre, 1000)
                            self.logger.info(f"Sub-enjambre {canal} ampliado: {sub_size} -> {len(sub_enjambre)}")

                    # Guardar estado en Redis
                    await self.redis.setex(f"swarm:{canal}:size", 300, json.dumps({"size": current_size, "sub_enjambres": len(self.sub_enjambres[canal])}))

            except Exception as e:
                self.logger.error(f"Error optimizando enjambres: {e}")
            await asyncio.sleep(300)

    async def partition_swarm(self, canal: str):
        enjambre = self.enjambres[canal]
        self.sub_enjambres[canal] = []
        for i in range(0, len(enjambre), self.max_sub_enjambre_size):
            self.sub_enjambres[canal].append(enjambre[i:i + self.max_sub_enjambre_size])
        self.logger.info(f"Enjambre {canal} particionado en {len(self.sub_enjambres[canal])} sub-enjambres")

    async def registrar_celu_entidad(self, celu: CeluEntidadCoreC):
        self.celu_entidades[celu.id] = celu
        self.logger.info(f"[ModuloRegistro] CeluEntidad {celu.id} registrada")
        if celu.canal in self.nucleus.canales_criticos:
            espejo_id = f"espejo_{celu.id}"
            procesador = celu.procesador
            espejo = CeluEntidadCoreC(
                espejo_id, procesador, celu.canal, celu.intervalo,
                self.nucleus.db_config, es_espejo=True, original_id=celu.id,
                instance_id=self.nucleus.instance_id
            )
            self.celu_entidades[espejo_id] = espejo
            self.logger.info(f"[ModuloRegistro] Espejo {espejo_id} registrado para {celu.id}")

    async def registrar_micro_celu_entidad(self, micro: MicroCeluEntidadCoreC):
        canal = micro.canal
        if len(self.enjambres[canal]) >= self.max_enjambre_por_canal:
            await self._limpiar_enjambre(canal)
        self.micro_celu_entidades[micro.id] = micro
        self.enjambres[canal].append(micro)

        # Asignar a sub-enjambre
        if not self.sub_enjambres[canal] or len(self.sub_enjambres[canal][-1]) >= self.max_sub_enjambre_size:
            self.sub_enjambres[canal].append([])
        self.sub_enjambres[canal][-1].append(micro)

        # Guardar estado en caché
        await self.redis.setex(f"micro_celu:{micro.id}", 3600, json.dumps({"canal": canal, "fitness": micro.dna.fitness}))
        self.logger.info(f"[ModuloRegistro] Micro-CeluEntidad {micro.id} registrada en enjambre {canal}")

    async def _limpiar_enjambre(self, canal: str):
        for sub_enjambre in self.sub_enjambres[canal]:
            await self._limpiar_sub_enjambre(canal, sub_enjambre)
        self.enjambres[canal] = [m for sub in self.sub_enjambres[canal] for m in sub]
        self.logger.info(f"[ModuloRegistro] Enjambre {canal} limpiado")

    async def _limpiar_sub_enjambre(self, canal: str, sub_enjambre: List[MicroCeluEntidadCoreC]):
        sub_enjambre.sort(key=lambda x: x.dna.fitness)
        for micro in sub_enjambre[:len(sub_enjambre)//2]:
            if micro.id in self.micro_celu_entidades:
                await micro.detener()
                del self.micro_celu_entidades[micro.id]
                await self.redis.delete(f"micro_celu:{micro.id}")
        sub_enjambre[:] = [m for m in sub_enjambre if m.id in self.micro_celu_entidades]

    async def regenerar_enjambre(self, canal: str, cantidad: int):
        await self.regenerar_sub_enjambre(canal, self.sub_enjambres[canal][-1] if self.sub_enjambres[canal] else [], cantidad)

    async def regenerar_sub_enjambre(self, canal: str, sub_enjambre: List[MicroCeluEntidadCoreC], cantidad: int):
        dna_base = MicroNanoDNA("calcular_valor", {"min": 0, "max": 1})
        for i in range(cantidad):
            try:
                dna = dna_base.heredar()
                async def funcion():
                    return {"valor": random.random()}
                micro = MicroCeluEntidadCoreC(
                    f"micro{time.time_ns()}_{i}_{self.nucleus.instance_id}",
                    funcion, canal, 0.1, self.nucleus.redis_client,
                    instance_id=self.nucleus.instance_id, dna=dna
                )
                await self.nucleus.registrar_micro_celu_entidad(micro)
                sub_enjambre.append(micro)
            except Exception as e:
                self.logger.error(f"[ModuloRegistro] Error regenerando micro-celu: {e}")
        self.logger.info(f"[ModuloRegistro] Sub-enjambre {canal} regenerado con {cantidad} micro-celus")

    async def ejecutar(self):
        pass

    async def detener(self):
        await self.redis.close()
        self.logger.info("[ModuloRegistro] Detenido")