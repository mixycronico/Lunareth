import asyncio
import json
import time
import psycopg2
import zstandard as zstd
import random
from ..utils.logging import logger
from .base import ModuloBase

class ModuloSincronizacion(ModuloBase):
    def __init__(self):
        self.logger = logger.getLogger("ModuloSincronizacion")
        self.migraciones_activas = 0
        self.max_migraciones = 10

    async def inicializar(self, nucleus: 'CoreCNucleus'):
        self.nucleus = nucleus
        self.redis = nucleus.redis_client
        self.stream_key = f"corec_stream_{self.nucleus.instance_id}"
        self.logger.info("[ModuloSincronizacion] Inicializado")

    async def ejecutar(self):
        async def limpiar_nodos():
            while True:
                try:
                    conn = psycopg2.connect(**self.nucleus.db_config)
                    cur = conn.cursor()
                    cur.execute("DELETE FROM nodos WHERE ultima_actividad < %s", (time.time() - 180,))
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    self.logger.error(f"[ModuloSincronizacion] Error limpiando nodos: {e}")
                await asyncio.sleep(60)

        async def balancear_carga():
            while True:
                try:
                    if self.migraciones_activas >= self.max_migraciones:
                        await asyncio.sleep(10)
                        continue
                    conn = psycopg2.connect(**self.nucleus.db_config)
                    cur = conn.cursor()
                    cur.execute("SELECT instance_id, AVG(carga) FROM nodos GROUP BY instance_id")
                    cargas = {row[0]: row[1] for row in cur.fetchall()}
                    cur.close()
                    conn.close()

                    if self.nucleus.instance_id in cargas and cargas[self.nucleus.instance_id] > 0.6:
                        canales = list(self.nucleus.modulo_registro.enjambres.keys())
                        if canales:
                            canal = random.choice(canales)
                            contexto = f"Balanceo de carga en CoreC, instancia {self.nucleus.instance_id}"
                            datos = {"cargas": cargas, "canales": canales}
                            analisis = await self.nucleus.razonar(datos, contexto)
                            if analisis["estado"] == "ok" and "canal" in analisis["respuesta"].lower():
                                suggested_canal = analisis["respuesta"].lower()
                                canal = next((c for c in canales if c in suggested_canal), canal)
                            destino = min(cargas, key=lambda k: cargas[k] if k != self.nucleus.instance_id else float('inf'))
                            self.migraciones_activas += 1
                            try:
                                await self.nucleus.modulo_registro.migrar_enjambre(canal, destino)
                            finally:
                                self.migraciones_activas -= 1
                except Exception as e:
                    self.logger.error(f"[ModuloSincronizacion] Error balanceando carga: {e}")
                await asyncio.sleep(300)

        await asyncio.gather(limpiar_nodos(), balancear_carga())

    async def detener(self):
        self.logger.info("[ModuloSincronizacion] Detenido")