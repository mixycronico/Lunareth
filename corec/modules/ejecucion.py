# corec/modules/ejecucion.py
import logging, asyncio
from typing import Dict, Any
import psycopg2

from corec.core import ModuloBase, celery_app
from corec.blocks import BloqueSimbiotico

class ModuloEjecucion(ModuloBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloEjecucion")
        self.nucleus = None

    async def inicializar(self, nucleus):
        self.nucleus = nucleus
        self.logger.info("[Ejecucion] listo")

    @celery_app.task(autoretry_for=(Exception,), retry_kwargs={'max_retries':3,'countdown':5})
    def ejecutar_bloque_task(bloque_id:str, db_config:Dict[str,Any]):
        try:
            bloque = BloqueSimbiotico(bloque_id,0,[],nucleus=None)
            asyncio.run(bloque.escribir_postgresql(db_config))
            logging.getLogger("ModuloEjecucion").info(f"[Ejecucion] bloque {bloque_id} ok")
        except Exception as e:
            logging.getLogger("ModuloEjecucion").error(f"[Ejecucion] err {e}")
            raise

    async def ejecutar(self):
        while True:
            try:
                regs = self.nucleus.modules["registro"].bloques
                for bid in regs:
                    self.ejecutar_bloque_task.delay(bid,self.nucleus.db_config)
                self.logger.info("[Ejecucion] tasks encoladas")
            except Exception as e:
                self.logger.error(f"[Ejecucion] err {e}")
            await asyncio.sleep(60)

    async def detener(self):
        self.logger.info("[Ejecucion] detenido")