import logging
from sklearn.ensemble import IsolationForest
import psycopg2
from corec.core import ComponenteBase


class ModuloAuditoria(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("ModuloAuditoria")
        self.nucleus = None
        self.detector = IsolationForest(contamination=0.1)

    async def inicializar(self, nucleus):
        """Inicializa el módulo de auditoría."""
        self.nucleus = nucleus
        self.logger.info("[Auditoria] Inicializado")

    async def detectar_anomalias(self):
        """Detecta anomalías en los datos de los bloques."""
        try:
            conn = psycopg2.connect(**self.nucleus.db_config)
            cur = conn.cursor()
            cur.execute("SELECT num_entidades, fitness FROM bloques")
            datos = cur.fetchall()
            if datos:
                predicciones = self.detector.fit_predict(datos)
                anomalias = [i for i, pred in enumerate(predicciones) if pred == -1]
                if anomalias:
                    cur.execute("SELECT id FROM bloques")
                    ids = cur.fetchall()
                    for idx in anomalias:
                        await self.nucleus.publicar_alerta({
                            "tipo": "anomalia_detectada",
                            "bloque_id": ids[idx][0],
                            "timestamp": time.time()
                        })
            cur.close()
            conn.close()
        except Exception as e:
            self.logger.error(f"[Auditoria] error: {e}")

    async def detener(self):
        """Detiene el módulo de auditoría."""
        self.logger.info("[Auditoria] Detenido")
