import logging
import random
from corec.core import ComponenteBase


class ModuloAuditoria(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloAuditoria")
        self.nucleus = None

    async def inicializar(self, nucleus, config=None):
        """Inicializa el módulo de auditoría."""
        try:
            self.nucleus = nucleus
            self.logger.info("[Auditoría] Módulo inicializado")
        except Exception as e:
            self.logger.error(f"[Auditoría] Error inesperado al inicializar: {e}")

    async def detectar_anomalias(self):
        """Detecta anomalías en los bloques registrados."""
        try:
            registro = self.nucleus.modules.get("registro")
            if not registro or not registro.bloques:
                self.logger.info("[Auditoría] No hay bloques para auditar")
                return
            for bloque_id, datos in registro.bloques.items():
                if datos["fitness"] < 0 or datos["num_entidades"] < 0:
                    await self.nucleus.publicar_alerta({
                        "tipo": "anomalia_detectada",
                        "bloque_id": bloque_id,
                        "fitness": datos["fitness"],
                        "num_entidades": datos["num_entidades"],
                        "timestamp": random.random()
                    })
                    self.logger.info(f"[Auditoría] Anomalía detectada en bloque {bloque_id}")
        except Exception as e:
            self.logger.error(f"[Auditoría] Error detectando anomalías: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_auditoria",
                "mensaje": str(e),
                "timestamp": random.random()
            })
            raise

    async def detener(self):
        """Detiene el módulo de auditoría."""
        self.logger.info("[Auditoría] Módulo detenido")
