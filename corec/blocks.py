import logging
import random
import time
from typing import List, Dict, Any


class BloqueSimbiotico:
    def __init__(self, id: str, canal: int, entidades: List, nucleus, max_size_mb: float = 10.0):
        self.id = id
        self.canal = canal
        self.entidades = entidades
        self.fitness = 0.0  # Corregido de 0.5 a 0.0
        self.mensajes: List[Dict[str, Any]] = []
        self.max_size_mb = max_size_mb
        self.logger = logging.getLogger("BloqueSimbiotico")
        self.nucleus = nucleus

    async def procesar(self, carga: float) -> Dict[str, Any]:
        """Procesa las entidades del bloque y calcula el fitness."""
        try:
            resultados = []
            for entidad in self.entidades:
                if entidad.estado == "activa":
                    resultado = await entidad.procesar(carga)
                    valor = resultado.get("valor", 0)
                    if isinstance(valor, (int, float)):
                        resultados.append(valor)
                    self.mensajes.append({
                        "entidad_id": entidad.id,
                        "canal": self.canal,
                        "valor": valor,
                        "timestamp": time.time()
                    })
            if resultados:
                self.fitness = sum(resultados) / len(resultados)
            self.logger.debug(f"[Bloque {self.id}] Procesado, fitness: {self.fitness:.2f}")
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_procesado",
                "bloque_id": self.id,
                "fitness": self.fitness,
                "timestamp": time.time()
            })
            return {
                "bloque_id": self.id,
                "mensajes": self.mensajes,
                "fitness": self.fitness
            }
        except Exception as e:
            self.logger.error(f"[Bloque {self.id}] Error procesando: {e}")
            return {
                "bloque_id": self.id,
                "mensajes": self.mensajes,
                "fitness": self.fitness,
                "error": str(e)
            }

    async def escribir_postgresql(self, conn):
        """Escribe los mensajes del bloque en PostgreSQL."""
        try:
            cur = conn.cursor()
            for mensaje in self.mensajes:
                cur.execute(
                    "INSERT INTO bloques (id, canal, num_entidades, fitness, timestamp) VALUES (%s, %s, %s, %s, %s)",
                    (self.id, self.canal, len(self.entidades), self.fitness, mensaje["timestamp"])
                )
            conn.commit()
            cur.close()
            self.mensajes = []
            self.logger.info(f"[Bloque {self.id}] Mensajes escritos en PostgreSQL")
            await self.nucleus.publicar_alerta({
                "tipo": "mensajes_escritos",
                "bloque_id": self.id,
                "num_mensajes": len(self.mensajes),
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.error(f"[Bloque {self.id}] Error escribiendo en PostgreSQL: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_escritura",
                "bloque_id": self.id,
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def reparar(self):
        """Repara entidades inactivas o corruptas."""
        try:
            for entidad in self.entidades:
                if entidad.estado != "activa":
                    entidad.estado = "activa"
                    entidad.funcion = lambda x: {"valor": random.uniform(0, 1)}
            self.logger.info(f"[Bloque {self.id}] Entidades reparadas")
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_reparado",
                "bloque_id": self.id,
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.error(f"[Bloque {self.id}] Error reparando: {e}")
