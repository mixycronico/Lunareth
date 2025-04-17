import asyncio
import json
import time
import psycopg2
import zstandard as zstd
from typing import Dict, Any, Optional
from ..utils.logging import logger
from .processors.base import ProcesadorBase

class CeluEntidadCoreC:
    def __init__(self, id: str, procesador: ProcesadorBase, canal: str, intervalo: float = 1.0, db_config: Dict[str, Any] = None, es_espejo: bool = False, original_id: Optional[str] = None, instance_id: str = "corec1"):
        self.id = id
        self.procesador = procesador
        self.canal = canal
        self.intervalo = intervalo
        self.db_config = db_config or {}
        self.activo = False
        self.es_espejo = es_espejo
        self.original_id = original_id
        self.instance_id = instance_id
        self.contexto = {"tipo": procesador.__class__.__name__.lower(), "canal": canal, "instance_id": instance_id, "nano_id": id}
        self.logger = logger.getLogger(f"CeluEntidad-{'Espejo-' if es_espejo else ''}{id}")
        self.semaphore = asyncio.Semaphore(500)

    async def inicializar(self):
        self.activo = True
        await self._actualizar_config()
        self.logger.info(f"[CeluEntidad {self.id}] Iniciada en canal {self.canal}{' (espejo)' if self.es_espejo else ''}")

    async def _actualizar_config(self):
        try:
            config_path = f"configs/core/corec_config_{self.instance_id}.json"
            with open(config_path, "r+") as f:
                config = json.load(f)
                config["celu_entidades"] = config.get("celu_entidades", {})
                config["celu_entidades"][self.id] = {
                    "canal": self.canal,
                    "intervalo": self.intervalo,
                    "tipo": self.contexto["tipo"],
                    "es_espejo": self.es_espejo,
                    "original_id": self.original_id,
                    "instance_id": self.instance_id
                }
                f.seek(0)
                json.dump(config, f, indent=4)
        except Exception as e:
            self.logger.error(f"[CeluEntidad {self.id}] Error actualizando config: {e}")

    async def obtener_datos(self) -> Any:
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute("SELECT datos FROM eventos WHERE canal = %s ORDER BY timestamp DESC LIMIT 100", (self.canal,))
            datos = [json.loads(zstd.decompress(row[0]).decode()) for row in cur.fetchall()]
            cur.close()
            conn.close()
            return {"valores": datos}
        except Exception as e:
            self.logger.error(f"[CeluEntidad {self.id}] Error obteniendo datos: {e}")
            return {"valores": []}

    async def procesar(self) -> Dict[str, Any]:
        async with self.semaphore:
            try:
                datos = await self.obtener_datos()
                resultado = await self.procesador.procesar(datos, self.contexto)
                return {
                    "estado": "ok",
                    "nano_id": self.id,
                    "canal": self.canal,
                    "resultado": resultado,
                    "timestamp": time.time(),
                    "es_espejo": self.es_espejo,
                    "original_id": self.original_id,
                    "instance_id": self.instance_id
                }
            except Exception as e:
                self.logger.error(f"[CeluEntidad {self.id}] Error procesando: {e}")
                return {"estado": "error", "mensaje": str(e), "nano_id": self.id}

    async def comunicar(self, resultado: Dict[str, Any]):
        if resultado["estado"] == "ok":
            try:
                conn = psycopg2.connect(**self.db_config)
                cur = conn.cursor()
                cur.execute("SELECT AVG(carga) FROM nodos WHERE instance_id = %s", (self.instance_id,))
                carga = cur.fetchone()[0] or 0
                cur.close()
                prioridad = 1.0 if self.canal in ["alertas", "reparadora_acciones", "seguridad_alertas"] else 0.7 * (1 - carga)
                if prioridad > 0.6:
                    datos_comprimidos = zstd.compress(json.dumps(resultado).encode())
                    cur = conn.cursor()
                    cur.execute(
                        "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES (%s, %s, %s, %s)",
                        (self.canal, datos_comprimidos, resultado["timestamp"], self.instance_id)
                    )
                    conn.commit()
                    cur.close()
                    self.logger.debug(f"[CeluEntidad {self.id}] Resultado publicado en {self.canal}")
                conn.close()
            except Exception as e:
                self.logger.error(f"[CeluEntidad {self.id}] Error comunicando: {e}")

    async def ejecutar(self):
        await self.inicializar()
        while self.activo:
            try:
                resultado = await self.procesar()
                await self.comunicar(resultado)
                if self.canal == "coordinacion_nodos" or self.es_espejo:
                    await self._enviar_heartbeat()
                await asyncio.sleep(self.intervalo)
            except Exception as e:
                self.logger.error(f"[CeluEntidad {self.id}] Error en ciclo: {e}")
                await asyncio.sleep(5)

    async def _enviar_heartbeat(self):
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO nodos (nodo_id, ultima_actividad, carga, es_espejo, original_id, instance_id) VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (nodo_id, instance_id) DO UPDATE SET ultima_actividad = %s, carga = %s, es_espejo = %s, original_id = %s, instance_id = %s",
                (self.id, time.time(), 0.5, self.es_espejo, self.original_id, self.instance_id, time.time(), 0.5, self.es_espejo, self.original_id, self.instance_id)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            self.logger.error(f"[CeluEntidad {self.id}] Error enviando heartbeat: {e}")

    async def detener(self):
        self.activo = False
        self.logger.info(f"[CeluEntidad {self.id}] Detenida")