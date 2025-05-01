import time
import asyncio
import psutil
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import asyncpg
from corec.entities import EntidadBase
from corec.entities_superpuestas import EntidadSuperpuesta
from corec.utils.quantization import escalar


class BloqueSimbiotico:
    def __init__(
        self,
        id: str,
        canal: int,
        entidades: List[EntidadBase],
        max_size_mb: float,
        nucleus,
        quantization_step: float,
        max_errores: float,
        max_concurrent_tasks: int = None,
        cpu_intensive: bool = False,
        mutacion: Dict[str, Any] = None,
        autorreplicacion: Dict[str, Any] = None
    ):
        """Bloque simbiótico que procesa entidades y gestiona datos."""
        self.nucleus = nucleus
        self.logger = nucleus.logger
        self.id = id
        self.canal = canal
        self.entidades = entidades
        self.max_size_mb = max_size_mb
        self.mensajes: List[Dict[str, Any]] = []
        self.fitness: float = 0.0
        self.fallos = 0
        self.quantization_step = quantization_step
        self.max_errores = max_errores
        self.max_concurrent_tasks = max_concurrent_tasks or nucleus.config.concurrent_tasks_max
        self.current_concurrent_tasks = self.max_concurrent_tasks
        self.cpu_intensive = cpu_intensive
        self.cpu_low_cycles = deque(maxlen=nucleus.config.cpu_stable_cycles)
        self.performance_history = deque(maxlen=nucleus.config.performance_history_size)
        self.increment_factor = nucleus.config.concurrent_tasks_increment_factor_default
        self.last_processing_time = 0.0
        self.mutacion = mutacion or {
            "enabled": False,
            "min_fitness": 0.3,
            "mutation_rate": 0.1,
            "ml_enabled": False
        }
        self.autorreplicacion = autorreplicacion or {
            "enabled": False,
            "max_entidades": 10000,
            "min_fitness_trigger": 0.2
        }

    async def _get_cpu_percent(self) -> float:
        """Obtiene el promedio de CPU basado en múltiples lecturas."""
        readings = []
        for _ in range(self.nucleus.config.cpu_readings):
            readings.append(psutil.cpu_percent())
            await asyncio.sleep(self.nucleus.config.cpu_reading_interval)
        avg_cpu = sum(readings) / len(readings)
        self.logger.debug(
            f"Bloque {self.id} CPU readings: {readings}, average: {avg_cpu:.1f}%"
        )
        return avg_cpu

    def _adjust_increment_factor(self, processing_time: float):
        """Ajusta el factor de incremento basado en el historial de rendimiento."""
        self.performance_history.append(processing_time)
        if len(self.performance_history) < self.nucleus.config.performance_history_size:
            return
        avg = sum(self.performance_history) / len(self.performance_history)
        if avg > self.nucleus.config.performance_threshold:
            self.increment_factor = max(
                self.nucleus.config.increment_factor_min,
                self.increment_factor * 0.95
            )
            self.logger.info(
                f"Bloque {self.id} rendimiento lento (avg={avg:.3f}s), "
                f"reduciendo increment_factor a {self.increment_factor:.3f}"
            )
        else:
            self.increment_factor = min(
                self.nucleus.config.increment_factor_max,
                self.increment_factor * 1.05
            )
            self.logger.info(
                f"Bloque {self.id} rendimiento bueno (avg={avg:.3f}s), "
                f"aumentando increment_factor a {self.increment_factor:.3f}"
            )

    def _adjust_concurrent_tasks(self, cpu_percent: float, ram_percent: float):
        """Ajusta dinámicamente tareas concurrentes según CPU y RAM."""
        overload = (
            cpu_percent > self.nucleus.config.cpu_autoadjust_threshold * 100 or
            ram_percent > self.nucleus.config.ram_autoadjust_threshold * 100
        )
        if overload:
            new_tasks = max(
                self.nucleus.config.concurrent_tasks_min,
                int(self.current_concurrent_tasks *
                    self.nucleus.config.concurrent_tasks_reduction_factor)
            )
            if new_tasks != self.current_concurrent_tasks:
                self.logger.info(
                    f"Bloque {self.id} reduciendo tareas de "
                    f"{self.current_concurrent_tasks} a {new_tasks} "
                    f"(CPU={cpu_percent:.1f}%, RAM={ram_percent:.1f}%)"
                )
                self.current_concurrent_tasks = new_tasks
                self.cpu_low_cycles.clear()
        else:
            is_low = cpu_percent < self.nucleus.config.cpu_autoadjust_threshold * 80
            self.cpu_low_cycles.append(is_low)
            if (
                len(self.cpu_low_cycles) == self.nucleus.config.cpu_stable_cycles
                and all(self.cpu_low_cycles)
            ):
                new_tasks = min(
                    self.max_concurrent_tasks,
                    int(self.current_concurrent_tasks * self.increment_factor)
                )
                if new_tasks != self.current_concurrent_tasks:
                    self.logger.info(
                        f"Bloque {self.id} incrementando tareas de "
                        f"{self.current_concurrent_tasks} a {new_tasks} "
                        f"(CPU={cpu_percent:.1f}%, RAM={ram_percent:.1f}%, "
                        f"inc_factor={self.increment_factor:.3f})"
                    )
                    self.current_concurrent_tasks = new_tasks
                    self.cpu_low_cycles.clear()

    async def procesar(self, carga: float) -> Dict[str, Any]:
        """Procesa entidades en paralelo, cuantiza valores y verifica recursos/fallos."""
        cpu = await self._get_cpu_percent()
        mem_mb = psutil.virtual_memory().used / (1024 * 1024)
        ram = (mem_mb /
               (psutil.virtual_memory().total / (1024 * 1024))) * 100
        if cpu > self.nucleus.config.alert_threshold * 100 or ram > self.nucleus.config.alert_threshold * 100:
            self.logger.warning(
                f"Bloque {self.id} recursos excedidos: CPU={cpu:.1f}%, RAM={ram:.1f}%"
            )
            await self.nucleus.publicar_alerta({
                "tipo": "alerta_recursos",
                "bloque_id": self.id,
                "cpu_percent": cpu,
                "ram_percent": ram,
                "timestamp": time.time()
            })
            return {"bloque_id": self.id, "mensajes": [], "fitness": 0.0}

        self._adjust_concurrent_tasks(cpu, ram)

        self.mensajes = []
        fit_total = 0.0
        count = 0
        total = len(self.entidades)

        async def _proc(ent: EntidadBase):
            try:
                if self.cpu_intensive:
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor() as pool:
                        msg = await loop.run_in_executor(
                            pool, lambda: ent.procesar(carga).result()
                        )
                else:
                    msg = await ent.procesar(carga)
                val = msg.get("valor")
                if not isinstance(val, (int, float)):
                    self.logger.warning(f"Bloque {self.id} valor inválido de {ent.id}")
                    return
                record = {
                    "entidad_id": ent.id,
                    "canal": self.canal,
                    "valor": escalar(val, self.quantization_step),
                    "clasificacion": msg.get("clasificacion", ""),
                    "probabilidad": msg.get("probabilidad", 0.0),
                    "timestamp": time.time(),
                    "roles": msg.get("roles", {})
                }
                self.mensajes.append(record)
                nonlocal fit_total, count
                fit_total += record["valor"]
                count += 1
            except Exception as e:
                self.fallos += 1
                self.logger.error(f"Bloque {self.id} error en {ent.id}: {e}")

        start = time.time()
        batch = self.current_concurrent_tasks
        for i in range(0, total, batch):
            chunk = self.entidades[i:i + batch]
            await asyncio.gather(*(_proc(e) for e in chunk))
        raw = (fit_total / count) if count else 0.0
        self.fitness = escalar(raw, self.quantization_step)
        duration = time.time() - start
        self.last_processing_time = duration
        self._adjust_increment_factor(duration)

        # mutación y autorreplicación omitidas (igual lógica que antes) ...

        if total and (self.fallos / total) > self.max_errores:
            self.logger.error(f"Bloque {self.id} fallos críticos: {self.fallos}/{total}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_critico_bloque",
                "bloque_id": self.id,
                "fallos": self.fallos,
                "total_entidades": total,
                "timestamp": time.time()
            })
            await self.reparar()

        await self.nucleus.publicar_alerta({
            "tipo": "bloque_procesado",
            "bloque_id": self.id,
            "num_mensajes": count,
            "fitness": self.fitness,
            "processing_time_s": duration,
            "concurrent_tasks": self.current_concurrent_tasks,
            "cpu_percent": cpu,
            "ram_percent": ram,
            "increment_factor": self.increment_factor,
            "num_entidades": len(self.entidades),
            "timestamp": time.time()
        })

        if self.nucleus.db_pool:
            try:
                async with self.nucleus.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        INSERT INTO bloques (id, canal, num_entidades, fitness, timestamp)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (id) DO UPDATE
                          SET canal = EXCLUDED.canal,
                              num_entidades = EXCLUDED.num_entidades,
                              fitness = EXCLUDED.fitness,
                              timestamp = EXCLUDED.timestamp
                        """,
                        self.id, self.canal, len(self.entidades),
                        self.fitness, time.time()
                    )
                self.logger.debug(f"Bloque {self.id} estado actualizado en PostgreSQL")
            except asyncpg.exceptions.ConnectionDoesNotExistError:
                self.logger.error(
                    f"Bloque {self.id} conexión a PostgreSQL perdida, reintentando"
                )
                self.nucleus.db_pool = await self.nucleus.init_postgresql(
                    self.nucleus.config.db_config.model_dump()
                )
            except Exception as e:
                self.logger.error(
                    f"Bloque {self.id} error actualizando estado en PostgreSQL: {e}"
                )
                await self.nucleus.publicar_alerta({
                    "tipo": "error_escritura_bloque",
                    "bloque_id": self.id,
                    "mensaje": str(e),
                    "timestamp": time.time()
                })

        return {"bloque_id": self.id, "mensajes": self.mensajes, "fitness": self.fitness}

    async def reparar(self):
        """Repara el bloque reactivando entidades inactivas o normalizando roles."""
        repaired = False
        for ent in self.entidades:
            try:
                if getattr(ent, "estado", None) == "inactiva":
                    ent.estado = "activa"
                    self.logger.info(f"Bloque {self.id} reactivada entidad {ent.id}")
                    repaired = True
                if isinstance(ent, EntidadSuperpuesta):
                    ent.normalizar_roles()
                    self.logger.info(f"Bloque {self.id} normalizó roles en {ent.id}")
                    repaired = True
            except Exception as e:
                self.logger.error(f"Bloque {self.id} error reparando {ent.id}: {e}")
        self.fallos = 0
        if repaired:
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_reparado",
                "bloque_id": self.id,
                "timestamp": time.time()
            })

    async def escribir_postgresql(self, conn):
        """Escribe todos los mensajes pendientes en PostgreSQL."""
        try:
            for msg in self.mensajes:
                await conn.execute(
                    """
                    INSERT INTO mensajes (
                      bloque_id, entidad_id, canal, valor,
                      clasificacion, probabilidad, timestamp, roles
                    ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                    """,
                    self.id,
                    msg["entidad_id"],
                    msg["canal"],
                    msg["valor"],
                    msg.get("clasificacion", ""),
                    msg.get("probabilidad", 0.0),
                    msg["timestamp"],
                    json.dumps(msg.get("roles", {}))
                )
            count = len(self.mensajes)
            self.mensajes = []
            self.logger.info(f"Bloque {self.id} escribió {count} mensajes en PostgreSQL")
            await self.nucleus.publicar_alerta({
                "tipo": "mensajes_escritos",
                "bloque_id": self.id,
                "num_mensajes": count,
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.error(f"Bloque {self.id} error escribiendo mensajes: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_escritura",
                "bloque_id": self.id,
                "mensaje": str(e),
                "timestamp": time.time()
            })
