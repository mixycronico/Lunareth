# corec/blocks.py
import logging
import time
import asyncio
from typing import List, Dict, Any
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from corec.entities import EntidadBase
from corec.entities_superpuestas import EntidadSuperpuesta
from corec.utils.quantization import escalar
from corec.config import (
    QUANTIZATION_STEP_DEFAULT,
    MAX_FALLOS_CRITICOS,
    ALERT_THRESHOLD,
    CPU_AUTOADJUST_THRESHOLD,
    RAM_AUTOADJUST_THRESHOLD,
    CONCURRENT_TASKS_MIN,
    CONCURRENT_TASKS_MAX,
    CONCURRENT_TASKS_REDUCTION_FACTOR,
    CONCURRENT_TASKS_INCREMENT_FACTOR_DEFAULT,
    CPU_STABLE_CYCLES,
    CPU_READINGS,
    CPU_READING_INTERVAL,
    PERFORMANCE_HISTORY_SIZE,
    PERFORMANCE_THRESHOLD,
    INCREMENT_FACTOR_MIN,
    INCREMENT_FACTOR_MAX
)
import psutil
import json

class BloqueSimbiotico:
    def __init__(
        self,
        id: str,
        canal: int,
        entidades: List[EntidadBase],
        max_size_mb: float,
        nucleus,
        quantization_step: float = QUANTIZATION_STEP_DEFAULT,
        max_errores: float = MAX_FALLOS_CRITICOS,
        max_concurrent_tasks: int = None,
        cpu_intensive: bool = False
    ):
        """
        Bloque simbiótico que procesa entidades y gestiona datos.

        Args:
            id (str): Identificador único.
            canal (int): Canal de comunicación.
            entidades (List[EntidadBase]): Lista de entidades.
            max_size_mb (float): Tamaño máximo en MB.
            nucleus: Instancia de CoreCNucleus.
            quantization_step (float): Paso de cuantización.
            max_errores (float): Umbral de errores para alertas críticas.
            max_concurrent_tasks (int, optional): Máximo número de tareas concurrentes.
            cpu_intensive (bool): Si True, usa ThreadPoolExecutor para procesamiento.
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
        self.quantization_step = quantization_step
        self.max_errores = max_errores
        self.max_concurrent_tasks = max_concurrent_tasks if max_concurrent_tasks is not None else CONCURRENT_TASKS_MAX
        self.current_concurrent_tasks = self.max_concurrent_tasks
        self.cpu_intensive = cpu_intensive
        self.cpu_low_cycles = deque(maxlen=CPU_STABLE_CYCLES)
        self.performance_history = deque(maxlen=PERFORMANCE_HISTORY_SIZE)  # Historial de tiempos
        self.increment_factor = CONCURRENT_TASKS_INCREMENT_FACTOR_DEFAULT

    async def _get_cpu_percent(self) -> float:
        """
        Obtiene el promedio de CPU basado en múltiples lecturas.

        Returns:
            float: Promedio del porcentaje de uso de CPU.
        """
        readings = []
        for _ in range(CPU_READINGS):
            readings.append(psutil.cpu_percent())
            await asyncio.sleep(CPU_READING_INTERVAL)
        avg_cpu = sum(readings) / len(readings)
        self.logger.debug(f"[Bloque {self.id}] CPU readings: {readings}, average: {avg_cpu:.1f}%")
        return avg_cpu

    def _adjust_increment_factor(self, processing_time: float):
        """
        Ajusta CONCURRENT_TASKS_INCREMENT_FACTOR basado en el historial de rendimiento.

        Args:
            processing_time (float): Tiempo de procesamiento del último ciclo (segundos).
        """
        self.performance_history.append(processing_time)
        if len(self.performance_history) < PERFORMANCE_HISTORY_SIZE:
            return  # Esperar a tener suficiente historial

        avg_processing_time = sum(self.performance_history) / len(self.performance_history)
        if avg_processing_time > PERFORMANCE_THRESHOLD:
            self.increment_factor = max(
                INCREMENT_FACTOR_MIN,
                self.increment_factor * 0.95
            )
            self.logger.info(
                f"[Bloque {self.id}] Rendimiento lento (avg={avg_processing_time:.3f}s), "
                f"reduciendo increment_factor a {self.increment_factor:.3f}"
            )
        else:
            self.increment_factor = min(
                INCREMENT_FACTOR_MAX,
                self.increment_factor * 1.05
            )
            self.logger.info(
                f"[Bloque {self.id}] Rendimiento bueno (avg={avg_processing_time:.3f}s), "
                f"aumentando increment_factor a {self.increment_factor:.3f}"
            )

    def _adjust_concurrent_tasks(self, cpu_percent: float, ram_percent: float):
        """
        Ajusta dinámicamente max_concurrent_tasks basado en CPU y RAM.

        Args:
            cpu_percent (float): Porcentaje de uso de CPU (promedio).
            ram_percent (float): Porcentaje de uso de RAM.
        """
        overload = (
            cpu_percent > CPU_AUTOADJUST_THRESHOLD * 100 or
            ram_percent > RAM_AUTOADJUST_THRESHOLD * 100
        )
        if overload:
            new_tasks = max(
                CONCURRENT_TASKS_MIN,
                int(self.current_concurrent_tasks * CONCURRENT_TASKS_REDUCTION_FACTOR)
            )
            if new_tasks != self.current_concurrent_tasks:
                self.logger.info(
                    f"[Bloque {self.id}] Reduciendo tareas concurrentes de {self.current_concurrent_tasks} a {new_tasks} "
                    f"(CPU={cpu_percent:.1f}%, RAM={ram_percent:.1f}%)"
                )
                self.current_concurrent_tasks = new_tasks
                self.cpu_low_cycles.clear()
        else:
            is_cpu_low = cpu_percent < CPU_AUTOADJUST_THRESHOLD * 80
            self.cpu_low_cycles.append(is_cpu_low)
            if len(self.cpu_low_cycles) == CPU_STABLE_CYCLES and all(self.cpu_low_cycles):
                new_tasks = min(
                    self.max_concurrent_tasks,
                    int(self.current_concurrent_tasks * self.increment_factor)
                )
                if new_tasks != self.current_concurrent_tasks:
                    self.logger.info(
                        f"[Bloque {self.id}] Incrementando tareas concurrentes de {self.current_concurrent_tasks} a {new_tasks} "
                        f"(CPU={cpu_percent:.1f}%, RAM={ram_percent:.1f}%, increment_factor={self.increment_factor:.3f})"
                    )
                    self.current_concurrent_tasks = new_tasks
                    self.cpu_low_cycles.clear()

    async def procesar(self, carga: float) -> Dict[str, Any]:
        """
        Procesa entidades en paralelo usando asyncio.gather o ThreadPoolExecutor, cuantiza valores y verifica recursos/fallos.

        Args:
            carga (float): Factor de carga (0.0 a 1.0).

        Returns:
            Dict[str, Any]: Resultado con mensajes, fitness y estado del bloque.
        """
        # Verificar recursos del sistema
        cpu_percent = await self._get_cpu_percent()
        mem_usage_mb = psutil.virtual_memory().used / (1024 * 1024)
        ram_percent = (mem_usage_mb / (psutil.virtual_memory().total / (1024 * 1024))) * 100
        if cpu_percent > ALERT_THRESHOLD * 100 or ram_percent > ALERT_THRESHOLD * 100:
            self.logger.warning(f"[Bloque {self.id}] Recursos excedidos: CPU={cpu_percent:.1f}%, RAM={ram_percent:.1f}%")
            await self.nucleus.publicar_alerta({
                "tipo": "alerta_recursos",
                "bloque_id": self.id,
                "cpu_percent": cpu_percent,
                "ram_percent": ram_percent,
                "timestamp": time.time()
            })
            return {"bloque_id": self.id, "mensajes": [], "fitness": 0.0}

        # Ajustar tareas concurrentes
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
                    self.logger.warning(f"[Bloque {self.id}] Valor inválido de {entidad.id}")
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
                self.logger.error(f"[Bloque {self.id}] Error en {entidad.id}: {e}")
                return None

        # Procesar entidades en lotes
        start_time = time.time()
        batch_size = self.current_concurrent_tasks
        for i in range(0, len(self.entidades), batch_size):
            batch = self.entidades[i:i + batch_size]
            resultados = await asyncio.gather(
                *(procesar_entidad(entidad) for entidad in batch),
                return_exceptions=True
            )
            for resultado in resultados:
                if isinstance(resultado, Exception):
                    self.fallos += 1
                    self.logger.error(f"[Bloque {self.id}] Excepción en procesamiento paralelo: {resultado}")
                    continue
                if resultado is not None:
                    self.mensajes.append(resultado)
                    fitness_total += resultado["valor"]
                    num_mensajes += 1

        # Calcular fitness y ajustar increment_factor
        raw_fit = (fitness_total / num_mensajes) if num_mensajes else 0.0
        self.fitness = escalar(raw_fit, self.quantization_step)
        processing_time = time.time() - start_time
        self._adjust_increment_factor(processing_time)

        # Mutar roles y crear nuevas entidades si fitness es bajo
        ml_module = self.nucleus.modules.get("ml")
        for entidad in self.entidades:
            if isinstance(entidad, EntidadSuperpuesta):
                entidad.mutar_roles(self.fitness, ml_module)
                # Crear nueva entidad si fitness es muy bajo
                if self.fitness < self.max_errores * 0.5 and len(self.entidades) < 10000:  # Límite de entidades
                    nueva_entidad = entidad.crear_entidad(self.id, self.canal)
                    self.entidades.append(nueva_entidad)
                    self.nucleus.entrelazador.registrar_entidad(nueva_entidad)
                    self.logger.info(f"[Bloque {self.id}] Nueva entidad creada: {nueva_entidad.id}")

        # Verificar fallos críticos
        if total_entidades > 0 and (self.fallos / total_entidades) > self.max_errores:
            self.logger.error(f"[Bloque {self.id}] Fallos críticos: {self.fallos}/{total_entidades}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_critico_bloque",
                "bloque_id": self.id,
                "fallos": self.fallos,
                "total_entidades": total_entidades,
                "timestamp": time.time()
            })
            await self.reparar()

        # Publicar alerta de procesamiento
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
            "timestamp": time.time()
        })

        return {
            "bloque_id": self.id,
            "mensajes": self.mensajes,
            "fitness": self.fitness
        }

    async def reparar(self):
        """Repara el bloque reactivando entidades inactivas o reiniciando roles."""
        for entidad in self.entidades:
            if getattr(entidad, "estado", None) == "inactiva":
                try:
                    entidad.estado = "activa"
                    self.logger.info(f"[Bloque {self.id}] Entidad {entidad.id} reactivada")
                except Exception as e:
                    self.logger.error(f"[Bloque {self.id}] Error al reactivar {entidad.id}: {e}")
            if isinstance(entidad, EntidadSuperpuesta):
                try:
                    entidad.normalizar_roles()
                    self.logger.info(f"[Bloque {self.id}] Roles de {entidad.id} normalizados")
                except Exception as e:
                    self.logger.error(f"[Bloque {self.id}] Error normalizando roles de {entidad.id}: {e}")
        self.fallos = 0
        await self.nucleus.publicar_alerta({
            "tipo": "bloque_reparado",
            "bloque_id": self.id,
            "timestamp": time.time()
        })

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
