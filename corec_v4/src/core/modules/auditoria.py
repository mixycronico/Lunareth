import asyncio
import time
import psycopg2
from ..utils.logging import logger
from .base import ModuloBase

class ModuloAuditoria(ModuloBase):
    def __init__(self):
        self.logger = logger.getLogger("ModuloAuditoria")

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.logger.info("[ModuloAuditoria] Inicializado")

    async def ejecutar(self):
        while True:
            try:
                conn = psycopg2.connect(**self.nucleus.db_config)
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM auditoria WHERE timestamp > %s AND tipo LIKE '%error%'", (time.time() - 3600,))
                errores = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT micro_id) FROM eventos WHERE dna->>'fitness' < '0.1' AND timestamp > %s", (time.time() - 3600,))
                micro_celus_debiles = cur.fetchone()[0]
                cur.close()
                conn.close()

                alerta = {
                    "tipo": "auditoria",
                    "valor": {"errores": errores, "micro_celus_debiles": micro_celus_debiles},
                    "nano_id": "auditoria",
                    "prioridad": 1
                }

                if errores > 100 or micro_celus_debiles > 100:
                    alerta["prioridad"] = 3
                    contexto = f"Auditoría del sistema CoreC, instancia {self.nucleus.instance_id}"
                    analisis = await self.nucleus.razonar(alerta["valor"], contexto)
                    alerta["analisis"] = analisis["respuesta"]

                await self.nucleus.publicar_alerta(alerta)
            except Exception as e:
                self.logger.error(f"[ModuloAuditoria] Error auditando: {e}")
            await asyncio.sleep(300)

    async def detener(self):
        self.logger.info("[ModuloAuditoria] Detenido")