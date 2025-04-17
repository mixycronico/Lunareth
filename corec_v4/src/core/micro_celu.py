import time
import torch
import random
import zstandard as zstd
import psycopg2
import json
from typing import Dict, Any, Callable
from ..utils.logging import logger
from .micro_nano_dna import MicroNanoDNA

class MicroCeluEntidadCoreC:
    def __init__(self, id: str, funcion: Callable[[], Any], canal: str, intervalo: float = 0.1, redis_client=None, instance_id: str = "corec1", dna: MicroNanoDNA = None):
        self.id = id
        self.funcion = funcion
        self.canal = canal
        self.intervalo = intervalo
        self.redis = redis_client
        self.instance_id = instance_id
        self.dna = dna or MicroNanoDNA("calcular_valor", {"min": 0, "max": 1})
        self.activo = False
        self.fallos = 0
        self.exitos = 0
        self.logger = logger.getLogger(f"MicroCeluEntidad-{id}")
        self.nucleus = None
        self._buffer = []
        self._metabolismo = 1.0

    async def procesar(self) -> Dict[str, Any]:
        try:
            resultado = await self.funcion()
            self.exitos += 1
            self.dna.fitness = self.exitos / (self.exitos + self.fallos + 1)
            inputs = torch.tensor([self.dna.fitness, self.exitos, self.fallos, random.random(), random.random(), random.random()], dtype=torch.float32)
            targets = torch.tensor([resultado["valor"], 1.0, 0.0], dtype=torch.float32)
            loss = self.dna.entrenar_red(inputs, targets)
            return {
                "estado": "ok",
                "micro_id": self.id,
                "canal": self.canal,
                "resultado": resultado,
                "timestamp": time.time(),
                "instance_id": self.instance_id,
                "dna": {"funcion_id": self.dna.funcion_id, "parametros": self.dna.parametros, "fitness": self.dna.fitness}
            }
        except Exception as e:
            self.fallos += 1
            self.dna.fitness = self.exitos / (self.exitos + self.fallos + 1)
            self.logger.error(f"[MicroCeluEntidad {self.id}] Error procesando: {e}")
            return {"estado": "error", "mensaje": str(e), "micro_id": self.id}

    async def comunicar(self, resultado: Dict[str, Any]):
        if resultado["estado"] == "ok" and self.redis:
            conn = psycopg2.connect(**self.nucleus.db_config)
            cur = conn.cursor()
            cur.execute("SELECT AVG(carga) FROM nodos WHERE instance_id = %s", (self.instance_id,))
            carga = cur.fetchone()[0] or 0
            cur.close()
            conn.close()
            self._metabolismo = self.dna.fitness * (1 - carga)
            if self._metabolismo > 0.6:
                self._buffer.append(resultado)
                if len(self._buffer) >= 10:
                    datos_comprimidos = zstd.compress(json.dumps(self._buffer).encode())
                    for _ in range(3):
                        try:
                            await self.redis.xadd(f"corec_stream_{self.instance_id}", {"data": datos_comprimidos})
                            break
                        except Exception:
                            await asyncio.sleep(0.1)
                    self._buffer.clear()
                    self.logger.debug(f"[MicroCeluEntidad {self.id}] Resultado publicado en {self.canal}")
            if self.fallos > 5:
                try:
                    await self.nucleus.modulo_registro.regenerar_enjambre(self.canal, 1)
                    await self.detener()
                except Exception as e:
                    self.logger.error(f"[MicroCeluEntidad {self.id}] Error regenerando: {e}")

    async def ejecutar(self):
        self.activo = True
        while self.activo:
            try:
                resultado = await self.procesar()
                await self.comunicar(resultado)
                await asyncio.sleep(self.intervalo)
            except Exception as e:
                self.logger.error(f"[MicroCeluEntidad {self.id}] Error en ciclo: {e}")
                await asyncio.sleep(1)

    async def detener(self):
        self.activo = False
        self.logger.info(f"[MicroCeluEntidad {self.id}] Detenida")