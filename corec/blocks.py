import logging
import time
from typing import List, Dict, Any
from corec.entities import Entidad

class BloqueSimbiotico:
    def __init__(self, id: str, canal: int, entidades: List[Entidad], max_size_mb: float, nucleus):
        """Un bloque simbiótico que procesa entidades y gestiona datos.

        Args:
            id (str): Identificador único del bloque.
            canal (int): Canal de comunicación del bloque.
            entidades (List[Entidad]): Lista de entidades que procesan datos.
            max_size_mb (float): Tamaño máximo del bloque en MB.
            nucleus: Instancia de CoreCNucleus para alertas y coordinación.
        """
        self.logger = logging.getLogger("BloqueSimbiotico")
        self.id = id
        self.canal = canal
        self.entidades = entidades
        self.max_size_mb = max_size_mb
        self.nucleus = nucleus
        self.mensajes: List[Dict[str, Any]] = []
        self.fitness: float = 0.0
        self.fallos = 0
        self.ia_timeout_seconds = None  # Tiempo de espera para IA por bloque

    async def procesar(self, carga: float) -> Dict[str, Any]:
        """Procesa las entidades del bloque con una carga dada.

        Args:
            carga (float): Valor de carga para el procesamiento (entre 0 y 1).

        Returns:
            Dict[str, Any]: Diccionario con el ID del bloque, mensajes generados y fitness.
        """
        self.mensajes = []
        fitness_total = 0.0
        num_mensajes = 0
        try:
            for entidad in self.entidades:
                try:
                    mensaje = await entidad.procesar(carga)
                    if isinstance(mensaje.get("valor"), (int, float)):
                        self.mensajes.append({
                            "entidad_id": entidad.id,
                            "canal": self.canal,
                            "valor": mensaje["valor"],
                            "clasificacion": mensaje.get("clasificacion", ""),
                            "probabilidad": mensaje.get("probabilidad", 0.0),
                            "timestamp": time.time()
                        })
                        fitness_total += mensaje["valor"]
                        num_mensajes += 1
                    else:
                        self.logger.warning(f"[Bloque {self.id}] Valor inválido de entidad {entidad.id}")
                except Exception as e:
                    self.fallos += 1
                    self.logger.error(f"[Bloque {self.id}] Error procesando entidad {entidad.id}: {e}")
            self.fitness = fitness_total / num_mensajes if num_mensajes > 0 else 0.0
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_procesado",
                "bloque_id": self.id,
                "num_mensajes": num_mensajes,
                "fitness": self.fitness,
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.error(f"[Bloque {self.id}] Error procesando: {e}")
        return {
            "bloque_id": self.id,
            "mensajes": self.mensajes,
            "fitness": self.fitness
        }

    async def reparar(self):
        """Repara el bloque reactivando entidades inactivas."""
        for entidad in self.entidades:
            if getattr(entidad, "estado", None) == "inactiva":
                try:
                    entidad.estado = "activa"
                    self.logger.info(f"[Bloque {self.id}] Entidad {entidad.id} reactivada")
                    self.fallos = 0
                    await self.nucleus.publicar_alerta({
                        "tipo": "bloque_reparado",
                        "bloque_id": self.id,
                        "timestamp": time.time()
                    })
                except Exception as e:
                    self.logger.error(f"[Bloque {self.id}] Error al reactivar entidad {entidad.id}: {str(e)}")
                    if self.nucleus:
                        await self.nucleus.publicar_alerta({
                            "tipo": "error_reparacion",
                            "bloque_id": self.id,
                            "mensaje": str(e),
                            "timestamp": time.time()
                        })
                    raise

    async def escribir_postgresql(self, conn):
        """Escribe los mensajes del bloque en PostgreSQL.

        Args:
            conn: Conexión asíncrona a PostgreSQL (asyncpg).
        """
        try:
            for mensaje in self.mensajes:
                await conn.execute(
                    """
                    INSERT INTO mensajes (
                        bloque_id, entidad_id, canal, valor, clasificacion, probabilidad, timestamp
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    self.id,
                    mensaje["entidad_id"],
                    mensaje["canal"],
                    mensaje["valor"],
                    mensaje.get("clasificacion", ""),
                    mensaje.get("probabilidad", 0.0),
                    mensaje["timestamp"]
                )
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
