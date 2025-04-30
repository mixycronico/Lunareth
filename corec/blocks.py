import time
import asyncio
import random
import psutil
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import networkx as nx
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
        """Bloque simbiótico que procesa entidades y gestiona datos.

        Args:
            id (str): Identificador único.
            canal (int): Canal de comunicación.
            entidades (List[EntidadBase]): Lista de entidades.
            max_size_mb (float): Tamaño máximo en MB.
            nucleus: Instancia de CoreCNucleus.
            quantization_step (float): Paso de cuantización.
            max_errores (float): Umbral de errores para alertas críticas.
            max_concurrent_tasks (int, optional): Máximo número de tareas concurrentes.
            cpu_intensive (bool): Si True, usa ThreadPoolExecutor.
            mutacion (Dict): Configuración de mutaciones.
            autorreplicacion (Dict): Configuración de autorreplicación.
        """
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
        self.logger.debug(f"Bloque {self.id} CPU readings: {readings}, average: {avg_cpu:.1f}%")
        return avg_cpu


    def _adjust_increment_factor(self, processing_time: float):
        """Ajusta el factor de incremento basado en el historial de rendimiento."""
        self.performance_history.append(processing_time)
        if len(self.performance_history) < self.nucleus.config.performance_history_size:
            return
        avg_processing_time = sum(self.performance_history) / len(self.performance_history)
        if avg_processing_time > self.nucleus.config.performance_threshold:
            self.increment_factor = max(
                self.nucleus.config.increment_factor_min,
                self.increment_factor * 0.95
            )
            self.logger.info(
                f"Bloque {self.id} rendimiento lento (avg={avg_processing_time:.3f}s), "
                f"reduciendo increment_factor a {self.increment_factor:.3f}"
            )
        else:
            self.increment_factor = min(
                self.nucleus.config.increment_factor_max,
                self.increment_factor * 1.05
            )
            self.logger.info(
                f"Bloque {self.id} rendimiento bueno (avg={avg_processing_time:.3f}s), "
                f"aumentando increment_factor a {self.increment_factor:.3f}"
            )


    def _adjust_concurrent_tasks(self, cpu_percent: float, ram_percent: float):
        """Ajusta dinámicamente max_concurrent_tasks basado en CPU y RAM."""
        overload = (
            cpu_percent > self.nucleus.config.cpu_autoadjust_threshold * 100 or
            ram_percent > self.nucleus.config.ram_autoadjust_threshold * 100
        )
        if overload:
            new_tasks = max(
                self.nucleus.config.concurrent_tasks_min,
                int(self.current_concurrent_tasks * self.nucleus.config.concurrent_tasks_reduction_factor)
            )
            if new_tasks != self.current_concurrent_tasks:
                self.logger.info(
                    f"Bloque {self.id} reduciendo tareas concurrentes de {self.current_concurrent_tasks} "
                    f"a {new_tasks} (CPU={cpu_percent:.1f}%, RAM={ram_percent:.1f}%)"
                )
                self.current_concurrent_tasks = new_tasks
                self.cpu_low_cycles.clear()
        else:
            is_cpu_low = cpu_percent < self.nucleus.config.cpu_autoadjust_threshold * 80
            self.cpu_low_cycles.append(is_cpu_low)
            if len(self.cpu_low_cycles) == self.nucleus.config.cpu_stable_cycles and all(self.cpu_low_cycles):
                new_tasks = min(self.max_concurrent_tasks, int(self.current_concurrent_tasks * self.increment_factor))
                if new_tasks != self.current_concurrent_tasks:
                    self.logger.info(
                        f"Bloque {self.id} incrementando tareas concurrentes de {self.current_concurrent_tasks} "
                        f"a {new_tasks} (CPU={cpu_percent:.1f}%, RAM={ram_percent:.1f}%, "
                        f"increment_factor={self.increment_factor:.3f})"
                    )
                    self.current_concurrent_tasks = new_tasks
                    self.cpu_low_cycles.clear()


    async def procesar(self, carga: float) -> Dict[str, Any]:
        """Procesa entidades en paralelo, cuantiza valores y verifica recursos/fallos.

        Args:
            carga (float): Factor de carga (0.0 a 1.0).

        Returns:
            Dict[str, Any]: Resultado con mensajes procesados y fitness.
        """
        cpu_percent = await self._get_cpu_percent()
        mem_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
        ram_percent = (mem_usage_mb / (psutil.virtual_memory().total / (1024 * 1024))) * 100
        if cpu_percent > self.nucleus.config.alert_threshold * 100 or ram_percent > self.nucleus.config.alert_threshold * 100:
            self.logger.warning(f"Bloque {self.id} recursos excedidos: CPU={cpu_percent:.1f}%, RAM={ram_percent:.1f}%")
            await self.nucleus.publicar_alerta({
                "tipo": "alerta_recursos",
                "bloque_id": self.id,
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent,
                "timestamp": time.time()
            })
            return {"bloque_id": self.id, "mensajes": [], "fitness": 0.0}

        self._adjust_concurrent_tasks(cpu_percent, ram_percent)

        self.mensajes = []
        fitness_total = 0.0
        num_mensajes = 0
        total_entidades = len(self.entidades)

        async def procesar_entidad(entidad: EntidadBase) -> Dict[str, Any]:
            try:
                if self.cpu_intensive:
                    loop = asyncio.get_running_loop()
                    with ThreadPoolExecutor() as pool:
                        msg = await loop.run_in_executor(pool, lambda: entidad.procesar(carga).result())
                else:
                    msg = await entidad.procesar(carga)
                if not isinstance(msg.get("valor"), (int, float)):
                    self.logger.warning(f"Bloque {self.id} valor inválido de {entidad.id}")
                    return None
                return {
                    "entidad_id": entidad.id,
                    "canal": self.canal,
                    "valor": escalar(msg["valor"], self.quantization_step),
                    "clasificacion": msg.get("clasificacion", ""),
                    "probabilidad": msg.get("probabilidad", 0.0),
                    "timestamp": time.time(),
                    "roles": msg.get("roles", {})
                }
            except Exception as e:
                self.fallos += 1
                self.logger.error(f"Bloque {self.id} error procesando entidad {entidad.id}: {e}")
                return None

        start_time = time.time()
        batch_size = self.current_concurrent_tasks
        for i in range(0, len(self.entidades), batch_size):
            batch = self.entidades[i:i + batch_size]
            resultados = await asyncio.gather(*(procesar_entidad(entidad) for entidad in batch), return_exceptions=True)
            for resultado in resultados:
                if isinstance(resultado, Exception):
                    self.fallos += 1
                    self.logger.error(f"Bloque {self.id} excepción en procesamiento paralelo: {resultado}")
                    continue
                if resultado is not None:
                    self.mensajes.append(resultado)
                    fitness_total += resultado["valor"]
                    num_mensajes += 1

        raw_fit = (fitness_total / num_mensajes) if num_mensajes else 0.0
        self.fitness = escalar(raw_fit, self.quantization_step)
        processing_time = time.time() - start_time
        self.last_processing_time = processing_time
        self._adjust_increment_factor(processing_time)

        if self.mutacion.get("enabled", False):
            ml_module = self.nucleus.modules.get("ml") if self.mutacion.get("ml_enabled", False) else None
            for entidad in self.entidades:
                if isinstance(entidad, EntidadSuperpuesta):
                    await entidad.mutar_roles(self.fitness, ml_module)

        if self.autorreplicacion.get("enabled", False) and self.fitness < self.autorreplicacion.get("min_fitness_trigger", 0.2):
            max_entidades = self.autorreplicacion.get("max_entidades", 10000)
            if len(self.entidades) < max_entidades:
                for entidad in self.entidades[:10]:
                    if isinstance(entidad, EntidadSuperpuesta):
                        nueva_entidad = await entidad.crear_entidad(self.id, self.canal, self.nucleus.db_pool)
                        self.entidades.append(nueva_entidad)
                        self.nucleus.entrelazador.registrar_entidad(nueva_entidad)
                        self.logger.info(f"Bloque {self.id} nueva entidad creada: {nueva_entidad.id} (fitness={self.fitness})")

        if total_entidades > 0 and (self.fallos / total_entidades) > self.max_errores:
            self.logger.error(f"Bloque {self.id} fallos críticos: {self.fallos}/{total_entidades}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_critico_bloque",
                "bloque_id": self.id,
                "fallos": self.fallos,
                "total_entidades": total_entidades,
                "timestamp": time.time()
            })
            await self.reparar()

        await self.nucleus.publicar_alerta({
            "tipo": "bloque_procesado",
            "bloque_id": self.id,
            "num_mensajes": num_mensajes,
            "fitness": self.fitness,
            "processing_time_s": processing_time,
            "concurrent_tasks": self.current_concurrent_tasks,
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
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
                        self.id,
                        self.canal,
                        len(self.entidades),
                        self.fitness,
                        time.time()
                    )
                self.logger.debug(f"Bloque {self.id} estado actualizado en PostgreSQL")
            except asyncpg.exceptions.ConnectionDoesNotExistError:
                self.logger.error(f"Bloque {self.id} conexión a PostgreSQL perdida, intentando reconectar")
                self.nucleus.db_pool = await self.nucleus.init_postgresql(self.nucleus.config.db_config.model_dump())
            except Exception as e:
                self.logger.error(f"Bloque {self.id} error actualizando estado en PostgreSQL: {e}")
                await self.nucleus.publicar_alerta({
                    "tipo": "error_escritura_bloque",
                    "bloque_id": self.id,
                    "mensaje": f"Error actualizando estado: {e}",
                    "timestamp": time.time()
                })

        return {
            "bloque_id": self.id,
            "mensajes": self.mensajes,
            "fitness": self.fitness
        }


    async def reparar(self):
        """Repara el bloque reactivando entidades inactivas o reiniciando roles."""
        repaired = False
        for entidad in self.entidades:
            try:
                if getattr(entidad, "estado", None) == "inactiva":
                    entidad.estado = "activa"
                    self.logger.info(f"Bloque {self.id} entidad {entidad.id} reactivada")
                    repaired = True
                if isinstance(entidad, EntidadSuperpuesta):
                    entidad.normalizar_roles()
                    self.logger.info(f"Bloque {self.id} roles de {entidad.id} normalizados")
                    repaired = True
            except Exception as e:
                self.logger.error(f"Bloque {self.id} error reparando entidad {entidad.id}: {e}")
        self.fallos = 0
        if repaired:
            await self.nucleus.publicar_alerta({
                "tipo": "bloque_reparado",
                "bloque_id": self.id,
                "timestamp": time.time()
            })
        else:
            self.logger.debug(f"Bloque {self.id} no requirió reparación")


    async def escribir_postgresql(self, conn):
        """Escribe mensajes en PostgreSQL, incluyendo roles si existen."""
        try:
            for mensaje in self.mensajes:
                await conn.execute(
                    """
                    INSERT INTO mensajes (
                        bloque_id, entidad_id, canal, valor, clasificacion, probabilidad, timestamp, roles
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    self.id,
                    mensaje["entidad_id"],
                    mensaje["canal"],
                    mensaje["valor"],
                    mensaje.get("clasificacion", ""),
                    mensaje.get("probabilidad", 0.0),
                    mensaje["timestamp"],
                    json.dumps(mensaje.get("roles", {}))
                )
            self.logger.info(f"Bloque {self.id} escribió {len(self.mensajes)} mensajes en PostgreSQL")
            self.mensajes = []
            await self.nucleus.publicar_alerta({
                "tipo": "mensajes_escritos",
                "bloque_id": self.id,
                "num_mensajes": len(self.mensajes),
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.error(f"Bloque {self.id} error escribiendo mensajes en PostgreSQL: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_escritura",
                "bloque_id": self.id,
                "mensaje": str(e),
                "timestamp": time.time()
            })
