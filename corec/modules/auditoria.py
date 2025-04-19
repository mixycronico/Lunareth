#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corec/modules/auditoria.py
Módulo de auditoría para detección de anomalías en CoreC.
"""
from corec.core import ModuloBase, asyncio, logging, psycopg2, time
from typing import Dict, Any

class ModuloAuditoria(ModuloBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloAuditoria")
        self.nucleus = None

    async def inicializar(self, nucleus):
        self.nucleus = nucleus
        self.logger.info("[ModuloAuditoria] Inicializado")

    async def detectar_anomalias(self):
        try:
            conn = psycopg2.connect(**self.nucleus.db_config)
            try:
                cur = conn.cursor()
                cur.execute(
                    "SELECT id, num_entidades, fitness FROM bloques WHERE timestamp > %s",
                    (time.time() - 3600,)
                )
                datos = [(row[1], row[2]) for row in cur.fetchall()]
                if datos:
                    anomalias = self.nucleus.anomaly_detector.fit_predict(datos)
                    cur.execute(
                        "SELECT id, num_entidades, fitness FROM bloques WHERE timestamp > %s",
                        (time.time() - 3600,)
                    )
                    for i, (bloque_id, _, _) in enumerate(cur.fetchall()):
                        if anomalias[i] == -1:
                            await self.nucleus.publicar_alerta({
                                "tipo": "anomalia_bloque",
                                "bloque_id": bloque_id,
                                "prioridad": 2,
                                "timestamp": time.time()
                            })
                            self.logger.info(f"Anomalía detectada en bloque {bloque_id}")
                cur.close()
            finally:
                conn.close()
        except Exception as e:
            self.logger.error(f"[ModuloAuditoria] Error detectando anomalías: {e}")

    async def ejecutar(self):
        while True:
            await self.detectar_anomalias()
            await asyncio.sleep(300)

    async def detener(self):
        self.logger.info("[ModuloAuditoria] Detenido")