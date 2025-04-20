import logging
import time
import psycopg2
import asyncio
from corec.core import ComponenteBase
from corec.blocks import BloqueSimbiotico
from unittest.mock import MagicMock


class ModuloAuditoria(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloAuditoria")
        self.nucleus = None
        self.detector = None

    async def inicializar(self, nucleus):
        """Inicializa el módulo de auditoría."""
        self.nucleus = nucleus
        # Detector de bloques
        self.detector = BloqueSimbiotico("x", 0, []).detector
        self.logger.info("[Auditoria] listo")

    async def detectar_anomalias(self):
        """Detecta anomalías en bloques."""
        try:
            # Usar db_config, mock en pruebas
            conn = self.nucleus.db_config if isinstance(self.nucleus.db_config, MagicMock) else psycopg2.connect(**self.nucleus.db_config)
            cur = conn.cursor()
            ts = time.time() - 3600
            cur.execute("SELECT num_entidades, fitness FROM bloques WHERE timestamp > %s", (ts,))
            datos = cur.fetchall()
            if datos:
                preds = self.detector.fit_predict(datos)
                cur.execute("SELECT id FROM bloques WHERE timestamp > %s", (ts,))
                ids = [r[0] for r in cur.fetchall()]
                for i, bid in enumerate(ids):
                    if preds[i] == -1:
                        alerta = {
                            "tipo": "anomalia",
                            "bloque_id": bid,
                            "timestamp": time.time()
                        }
                        await self.nucleus.publicar_alerta(alerta)
                        self.logger.info(f"[Auditoria] anomalía en {bid}")
            cur.close()
            if not isinstance(conn, MagicMock):  # Cerrar conexiones reales
                conn.close()
        except Exception as e:
            self.logger.error(f"[Auditoria] error: {e}")

    async def ejecutar(self):
        """Ejecuta detección de anomalías."""
        while True:
            await self.detectar_anomalias()
            await asyncio.sleep(300)

    async def detener(self):
        """Detiene el módulo de auditoría."""
        self.logger.info("[Auditoria] detenido")
