# corec/blocks.py
import logging
import time
import statistics
import json
import zstd
import psycopg2
import asyncio
from typing import Dict, Any, List

from sklearn.ensemble import IsolationForest
from corec.entities import MicroCeluEntidadCoreC, crear_entidad, procesar_entidad
from corec.serialization import deserializar_mensaje

class BloqueSimbiotico:
    def __init__(
        self,
        id: str,
        canal: int,
        entidades: List[MicroCeluEntidadCoreC],
        max_size_mb: int = 1,
        nucleus=None
    ):
        self.id         = id
        self.canal      = canal
        self.entidades  = entidades
        self.fitness    = 0.0
        self.nucleus    = nucleus
        self.logger     = logging.getLogger(f"BloqueSimbiotico-{id}")
        self.detector   = IsolationForest(contamination=0.05)
        self.mensajes   = []
        self.umbral     = 0.5
        self.fallos     = 0

    async def ajustar_umbral(self, carga: float, valores: List[float], errores: int):
        try:
            desv = statistics.stdev(valores) if len(valores)>1 else 0.1
            nuevo = 0.5*carga + 0.3*desv + 0.2*(errores/max(1,len(self.entidades)))
            self.umbral = max(0.1, min(nuevo, 0.9))
            self.logger.info(f"[{self.id}] Umbral → {self.umbral:.2f}")
        except Exception as e:
            self.logger.error(f"[{self.id}] Error ajustando umbral: {e}")

    async def procesar(self, carga: float) -> Dict[str, Any]:
        n     = max(1, int(len(self.entidades)*min(carga,1.0)))
        subs  = self.entidades[:n]
        tareas= [procesar_entidad(e,self.umbral) for e in subs]
        res   = await asyncio.gather(*tareas, return_exceptions=True)
        msgs, err, vals = [], 0, []
        for r in res:
            if isinstance(r, Exception):
                err+=1; continue
            m = await deserializar_mensaje(r)
            msgs.append(m)
            if not m["activo"]: err+=1
            if m["valor"]>0:  vals.append(m["valor"])
        self.fitness = max(0.0, self.fitness - err/max(1,n))
        self.mensajes.extend(msgs)
        await self.ajustar_umbral(carga, vals, err)
        if err> n*0.05:
            await self.reparar(err)
        return {"bloque_id":self.id, "mensajes":msgs, "fitness":self.fitness}

    async def reparar(self, errores:int):
        for i,msg in enumerate(self.mensajes):
            if not msg["activo"]:
                self.fallos+=1
                if self.fallos>=2:
                    self.entidades[i]=crear_entidad(f"m{time.time_ns()}",self.canal,self.entidades[i][2])
                    self.fallos=0
        self.logger.info(f"[{self.id}] Reparado ({errores} errores)")
        self.mensajes.clear()

    async def escribir_postgresql(self, db_config: Dict[str,Any]):
        out = await self.procesar(0.5)
        try:
            conn = psycopg2.connect(**db_config)
            cur  = conn.cursor()
            comp = zstd.compress(json.dumps(out["mensajes"]).encode(), level=3)
            cur.execute(
                "INSERT INTO bloques (id, canal, num_entidades, fitness, timestamp, instance_id) "
                "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO UPDATE "
                "SET num_entidades=EXCLUDED.num_entidades, fitness=EXCLUDED.fitness, timestamp=EXCLUDED.timestamp",
                (self.id,self.canal,len(self.entidades),self.fitness,time.time(),self.nucleus.instance_id)
            )
            conn.commit()
            cur.close(); conn.close()
            self.mensajes.clear()
        except Exception as e:
            self.logger.error(f"[{self.id}] Error PG: {e}")