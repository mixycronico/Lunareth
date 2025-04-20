# corec/modules/auditoria.py
import logging, time, psycopg2, asyncio
from typing import Dict, Any

from corec.core import ModuloBase
from corec.blocks import BloqueSimbiotico

class ModuloAuditoria(ModuloBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloAuditoria")
        self.nucleus = None
        self.detector = None

    async def inicializar(self, nucleus):
        self.nucleus  = nucleus
        # tomamos el detector de bloques
        self.detector = BloqueSimbiotico("x",0,[]).detector
        self.logger.info("[Auditoria] listo")

    async def detectar_anomalias(self):
        try:
            conn = psycopg2.connect(**self.nucleus.db_config)
            cur  = conn.cursor()
            ts   = time.time()-3600
            cur.execute("SELECT num_entidades, fitness FROM bloques WHERE timestamp>%s",(ts,))
            datos = cur.fetchall()
            if datos:
                preds = self.detector.fit_predict(datos)
                cur.execute("SELECT id FROM bloques WHERE timestamp>%s",(ts,))
                ids = [r[0] for r in cur.fetchall()]
                for i,bid in enumerate(ids):
                    if preds[i]==-1:
                        alerta = {"tipo":"anomalia","bloque_id":bid,"timestamp":time.time()}
                        await self.nucleus.publicar_alerta(alerta)
                        self.logger.info(f"[Auditoria] anomalía en {bid}")
            cur.close(); conn.close()
        except Exception as e:
            self.logger.error(f"[Auditoria] error: {e}")

    async def ejecutar(self):
        while True:
            await self.detectar_anomalias()
            await asyncio.sleep(300)

    async def detener(self):
        self.logger.info("[Auditoria] detenido")