import logging
import time
from corec.core import ComponenteBase


class ModuloAuditoria(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloAuditoria")
        self.nucleus = None

    async def inicializar(self, nucleus, config=None):
        """Inicializa el módulo de auditoría.

        Args:
            nucleus: Instancia del núcleo de CoreC.
            config: Configuración del módulo (opcional).
        """
        try:
            self.nucleus = nucleus
            self.logger.info("Módulo Auditoría inicializado")
        except Exception as e:
            self.logger.error(f"Error inicializando Módulo Auditoría: {e}")
            raise

    async def detectar_anomalias(self):
        """Detecta anomalías en los bloques registrados."""
        try:
            registro = self.nucleus.modules.get("registro")
            if not registro or not registro.bloques:
                self.logger.info("No hay bloques para auditar")
                return
            for bloque_id, datos in registro.bloques.items():
                if datos["fitness"] < 0 or datos["num_entidades"] < 0:
                    await self.nucleus.publicar_alerta({
                        "tipo": "anomalia_detectada",
                        "bloque_id": bloque_id,
                        "fitness": datos["fitness"],
                        "num_entidades": datos["num_entidades"],
                        "timestamp": time.time()
                    })
                    self.logger.info(f"Anomalía detectada en bloque {bloque_id}")
        except Exception as e:
            self.logger.error(f"Error detectando anomalías: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_auditoria",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def detener(self):
        """Detiene el módulo de auditoría."""
        self.logger.info("Módulo Auditoría detenido")
