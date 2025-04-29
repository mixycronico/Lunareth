# corec/nucleus.py
import asyncio
import logging
import time
import json
from pathlib import Path
import asyncpg
import aioredis
import pandas as pd
from corec.config_loader import load_config_dict
from corec.scheduler import Scheduler
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from corec.modules.ia import ModuloIA
from corec.modules.analisis_datos import ModuloAnalisisDatos
from corec.modules.evolucion import ModuloEvolucion
from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad
from corec.entities_superpuestas import EntidadSuperpuesta
from corec.entrelazador import Entrelazador
from corec.utils.db_utils import init_postgresql, init_redis
from corec.config import QUANTIZATION_STEP_DEFAULT

class CoreCNucleus:
    def __init__(self, config_path: str):
        """Inicializa el núcleo de CoreC."""
        self.logger = logging.getLogger("CoreCNucleus")
        self.config_path = config_path
        self.config = None
        self.db_pool = None
        self.redis_client = None
        self.modules = {}
        self.plugins = {}
        self.bloques = []
        self.scheduler = None
        self.entrelazador = None
        self.fallback_storage = Path("fallback_messages.json")

    async def inicializar(self):
        """Configura conexiones, módulos, bloques, entrelazador y tareas."""
        try:
            self.config = load_config_dict(self.config_path)
            try:
                self.db_pool = await init_postgresql(self.config.get("db_config", {}))
            except Exception as e:
                self.logger.error(f"[Núcleo] Fallo en conexión a PostgreSQL tras reintentos: {e}")
                self.db_pool = None
                await self.publicar_alerta({
                    "tipo": "error_conexion_postgresql_critico",
                    "mensaje": str(e),
                    "timestamp": time.time()
                })

            try:
                self.redis_client = await init_redis(self.config.get("redis_config", {}))
            except Exception as e:
                self.logger.error(f"[Núcleo] Fallo en conexión a Redis tras reintentos: {e}")
                self.redis_client = None
                await self.publicar_alerta({
                    "tipo": "error_conexion_redis_critico",
                    "mensaje": str(e),
                    "timestamp": time.time()
                })

            # Inicializar Entrelazador
            self.entrelazador = Entrelazador(self.redis_client)
            self.logger.info("[Núcleo] Entrelazador inicializado")

            # Inicializar módulos
            self.modules["registro"] = ModuloRegistro()
            self.modules["sincronizacion"] = ModuloSincronizacion()
            self.modules["ejecucion"] = ModuloEjecucion()
            self.modules["auditoria"] = ModuloAuditoria()
            self.modules["ia"] = ModuloIA()
            self.modules["analisis_datos"] = ModuloAnalisisDatos()
            self.modules["evolucion"] = ModuloEvolucion()

            for name, module in self.modules.items():
                module_config = self.config.get(f"{name}_config", {})
                if name == "ia" and not module_config.get("enabled", True):
                    self.logger.info(f"[Núcleo] Módulo {name} deshabilitado")
                    continue
                try:
                    self.logger.debug(f"[Núcleo] Inicializando módulo {name}")
                    await module.inicializar(self, module_config)
                except Exception as e:
                    self.logger.error(f"[Núcleo] Error inicializando módulo {name}: {e}")
                    await self.publicar_alerta({
                        "tipo": f"error_inicializacion_{name}",
                        "mensaje": str(e),
                        "timestamp": time.time()
                    })

            # Inicializar bloques
            for block_conf in self.config.get("bloques", []):
                quantization_step = block_conf.get("quantization_step", QUANTIZATION_STEP_DEFAULT)
                max_errores = block_conf.get("autoreparacion", {}).get("max_errores", 0.1)
                max_concurrent_tasks = block_conf.get("max_concurrent_tasks")
                cpu_intensive = block_conf.get("cpu_intensive", False)
                entidades = []
                ia_config = self.config.get("ia_config", {})
                if block_conf["id"] == "ia_analisis" and ia_config.get("enabled", True):
                    async def ia_fn(carga, mod_ia=self.modules["ia"], bconf=block_conf):
                        datos = await self.get_datos_from_redis(bconf["id"])
                        res = await mod_ia.procesar_bloque(None, datos)
                        return res["mensajes"][0] if res["mensajes"] else {"valor": 0.0}
                    entidades = [
                        Entidad(f"ent_{i}", block_conf["canal"], ia_fn, quantization_step)
                        for i in range(block_conf["entidades"])
                    ]
                else:
                    entidades = [
                        EntidadSuperpuesta(
                            f"ent_{i}",
                            {"rol1": 0.5, "rol2": 0.5},  # Roles iniciales
                            quantization_step,
                            min_fitness=block_conf.get("autoreparacion", {}).get("min_fitness", 0.3)
                        )
                        for i in range(block_conf["entidades"])
                    ]
                bloque = BloqueSimbiotico(
                    block_conf["id"],
                    block_conf["canal"],
                    entidades,
                    block_conf["max_size_mb"],
                    self,
                    quantization_step,
                    max_errores,
                    max_concurrent_tasks,
                    cpu_intensive
                )
                self.bloques.append(bloque)
                for entidad in entidades:
                    self.entrelazador.registrar_entidad(entidad)
                await self.modules["registro"].registrar_bloque(
                    bloque.id, bloque.canal, len(entidades), bloque.max_size_mb
                )

            # Configurar enlaces
            for bloque in self.bloques:
                entidades = bloque.entidades
                for i in range(0, len(entidades), 2):
                    if i + 1 < len(entidades):
                        try:
                            self.entrelazador.enlazar(entidades[i], entidades[i + 1])
                        except ValueError as e:
                            self.logger.warning(f"[Núcleo] Error enlazando entidades en {bloque.id}: {e}")

            # Configurar scheduler
            self.scheduler = Scheduler()
            self.scheduler.start()
            for b in self.bloques:
                self.scheduler.schedule_periodic(
                    func=self.process_bloque,
                    seconds=60,
                    job_id=f"proc_{b.id}",
                    args=[b]
                )
            self.scheduler.schedule_periodic(
                func=self.modules["auditoria"].detectar_anomalias,
                seconds=120,
                job_id="audit_anomalias"
            )
            self.scheduler.schedule_periodic(
                func=self.ejecutar_analisis,
                seconds=300,
                job_id="analisis_datos_periodico"
            )
            self.scheduler.schedule_periodic(
                func=self.retry_fallback_messages,
                seconds=300,
                job_id="retry_fallback_messages"
            )
            self.scheduler.schedule_periodic(
                func=self.procesar_entrelazador,
                seconds=30,
                job_id="entrelazador_periodico"
            )
            self.scheduler.schedule_periodic(
                func=self.evaluar_estrategias,
                seconds=300,
                job_id="evolucion_periodica"
            )
            self.scheduler.schedule_periodic(
                func=self.consumir_aprendizajes,
                seconds=600,
                job_id="aprendizajes_periodica"
            )

            self.logger.info("[Núcleo] Inicialización completa")

        except FileNotFoundError as e:
            self.logger.error(f"[Núcleo] Configuración no encontrada: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"[Núcleo] Configuración inválida: {e}")
            raise
        except Exception as e:
            self.logger.error(f"[Núcleo] Error inicializando: {e}")
            await self.publicar_alerta({
                "tipo": "error_inicializacion",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def procesar_entrelazador(self):
        """Procesa cambios periódicos en el Entrelazador."""
        try:
            for bloque in self.bloques:
                cambio = {"fitness": bloque.fitness}
                for entidad in bloque.entidades:
                    await self.entrelazador.afectar(entidad, cambio, max_saltos=1)
            self.logger.debug("[Núcleo] Cambios propagados en Entrelazador")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error procesando Entrelazador: {e}")
            await self.publicar_alerta({
                "tipo": "error_entrelazador",
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def process_bloque(self, bloque: BloqueSimbiotico):
        """Procesa un bloque y escribe sus mensajes."""
        try:
            if bloque.id != "ia_analisis":
                await self.modules["ejecucion"].encolar_bloque(bloque)
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await bloque.escribir_postgresql(conn)
            else:
                self.logger.warning(f"[Núcleo] PostgreSQL no disponible, usando fallback para {bloque.id}")
                await self.save_fallback_messages(bloque.id, bloque.mensajes)
                await self.publicar_alerta({
                    "tipo": "error_db_pool",
                    "bloque_id": bloque.id,
                    "mensaje": "db_pool no inicializado, usando fallback",
                    "timestamp": time.time()
                })
        except Exception as e:
            self.logger.error(f"[Núcleo] Error procesando bloque {bloque.id}: {e}")
            await self.publicar_alerta({
                "tipo": "error_procesamiento_bloque",
                "bloque_id": bloque.id,
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def save_fallback_messages(self, bloque_id: str, mensajes: list):
        """Guarda mensajes en archivo local si PostgreSQL no está disponible."""
        try:
            if self.fallback_storage.exists():
                with open(self.fallback_storage, "r") as f:
                    existing = json.load(f)
            else:
                existing = []
            existing.extend([{"bloque_id": bloque_id, "mensaje": m} for m in mensajes])
            with open(self.fallback_storage, "w") as f:
                json.dump(existing, f)
            self.logger.info(f"[Núcleo] Mensajes de {bloque_id} guardados en fallback")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error guardando mensajes en fallback: {e}")

    async def retry_fallback_messages(self):
        """Reintenta escribir mensajes guardados en fallback a PostgreSQL."""
        if not self.db_pool or not self.fallback_storage.exists():
            self.logger.info(f"[Núcleo] No hay mensajes de fallback o db_pool no disponible")
            return
        try:
            self.logger.info(f"[Núcleo] Leyendo mensajes de {self.fallback_storage}")
            with open(self.fallback_storage, "r") as f:
                messages = json.load(f)
            if not isinstance(messages, list):
                self.logger.error(f"[Núcleo] Formato inválido en {self.fallback_storage}, esperado lista")
                return
            self.logger.info(f"[Núcleo] Encontrados {len(messages)} mensajes para reintentar")
            async with self.db_pool.acquire() as conn:
                self.logger.debug("[Núcleo] Adquirida conexión al pool de PostgreSQL")
                for msg in messages:
                    bloque_id = msg.get("bloque_id")
                    m = msg.get("mensaje")
                    if not (bloque_id and m):
                        self.logger.warning(f"[Núcleo] Mensaje inválido: {msg}")
                        continue
                    self.logger.debug(f"[Núcleo] Insertando mensaje para bloque {bloque_id}, entidad {m['entidad_id']}")
                    await conn.execute(
                        """
                        INSERT INTO mensajes (
                            bloque_id, entidad_id, canal, valor, clasificacion, probabilidad, timestamp
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                        """,
                        bloque_id,
                        m["entidad_id"],
                        m["canal"],
                        m["valor"],
                        m.get("clasificacion", ""),
                        m.get("probabilidad", 0.0),
                        m["timestamp"]
                    )
            self.logger.info(f"[Núcleo] Eliminando archivo de fallback {self.fallback_storage}")
            try:
                self.fallback_storage.unlink()
                self.logger.info("[Núcleo] Mensajes de fallback escritos en PostgreSQL")
            except Exception as e:
                self.logger.error(f"[Núcleo] Error eliminando archivo de fallback: {e}")
                self.logger.debug(f"[Núcleo] Permisos del archivo: {self.fallback_storage.stat()}")
        except json.JSONDecodeError as e:
            self.logger.error(f"[Núcleo] Error decodificando JSON en {self.fallback_storage}: {e}")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error reintentando mensajes de fallback: {e}")
            self.logger.debug(f"[Núcleo] Detalles del error: {repr(e)}")

    async def ejecutar_analisis(self):
        """Extrae mensajes de PostgreSQL, crea un DataFrame y ejecuta análisis."""
        try:
            if not self.db_pool:
                self.logger.warning("[Núcleo] No se puede ejecutar análisis, db_pool no inicializado")
                return
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT bloque_id, valor FROM mensajes LIMIT 1000")
            if not rows:
                self.logger.debug("[Núcleo] No hay mensajes para análisis")
                return
            df = pd.DataFrame(rows, columns=["bloque_id", "valor"])
            df_wide = df.pivot(columns="bloque_id", values="valor").fillna(0)
            await self.modules["analisis_datos"].analizar(df_wide, "mensajes_recientes")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error en ejecutar_analisis: {e}")
            await self.publicar_alerta({
                "tipo": "error_analisis",
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def get_datos_from_redis(self, bloque_id: str) -> dict:
        """Lee mensajes de Redis para alimentar al módulo IA."""
        try:
            if not self.redis_client:
                self.logger.warning(f"[Núcleo] Redis no inicializado para {bloque_id}")
                return {"valores": []}
            max_length = self.config.get("redis_config", {}).get("stream_max_length", 5000)
            msgs = await self.redis_client.xread({"alertas:datos": "$"}, block=1000, count=10)
            valores = []
            for _, batch in msgs:
                for _, msg in batch:
                    if msg.get("bloque_id") in ["enjambre_sensor", "crypto_trading", "nodo_seguridad"]:
                        valores.extend(msg.get("valores", []))
            if len(valores) > max_length:
                valores = valores[-max_length:]
            return {"valores": valores}
        except aioredis.RedisError as e:
            self.logger.error(f"[Núcleo] Error leyendo Redis para {bloque_id}: {e}")
            await self.publicar_alerta({
                "tipo": "error_redis_datos",
                "bloque_id": bloque_id,
                "mensaje": str(e),
                "timestamp": time.time()
            })
            return {"valores": []}

    async def publicar_alerta(self, alerta: dict):
        """Publica una alerta en Redis y la archiva en PostgreSQL si es necesario."""
        try:
            if not self.redis_client:
                self.logger.warning("[Alerta] Redis no inicializado, archivando localmente")
                await self.archive_alert(alerta)
                return
            key = f"alertas:{alerta['tipo']}"
            max_length = self.config.get("redis_config", {}).get("stream_max_length", 5000)
            await self.redis_client.xadd(key, alerta, maxlen=max_length)
            self.logger.debug(f"[Alerta] {alerta}")
            stream_len = await self.redis_client.xlen(key)
            if stream_len >= max_length * 0.9:
                await self.archive_alert(alerta)
        except aioredis.RedisError as e:
            self.logger.error(f"[Núcleo] Error publicando alerta: {e}")
            await self.archive_alert(alerta)

    async def archive_alert(self, alerta: dict):
        """Archiva una alerta en PostgreSQL."""
        if not self.db_pool:
            self.logger.warning("[Alerta] PostgreSQL no disponible, no se puede archivar")
            return
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO alertas (tipo, bloque_id, mensaje, timestamp, datos)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    alerta.get("tipo", "unknown"),
                    alerta.get("bloque_id", ""),
                    alerta.get("mensaje", ""),
                    alerta.get("timestamp", time.time()),
                    json.dumps(alerta)
                )
            self.logger.info(f"[Alerta] Alerta archivada en PostgreSQL: {alerta['tipo']}")
        except Exception as e:
            self.logger.error(f"[Alerta] Error archivando alerta: {e}")

    async def publicar_aprendizaje(self, aprendizaje: dict):
        """Publica un aprendizaje en Redis para compartir con otros núcleos."""
        try:
            if not self.redis_client:
                self.logger.warning("[Aprendizaje] Redis no inicializado")
                return
            aprendizaje["instancia_id"] = self.config["instance_id"]
            aprendizaje["timestamp"] = time.time()
            await self.redis_client.xadd("corec:aprendizajes", aprendizaje, maxlen=1000)
            self.logger.info(f"[Aprendizaje] Publicado: {aprendizaje}")
            # Archivar en PostgreSQL
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO aprendizajes (instancia_id, bloque_id, estrategia, fitness, timestamp)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    aprendizaje["instancia_id"],
                    aprendizaje.get("bloque_id", ""),
                    json.dumps(aprendizaje["estrategia"]),
                    aprendizaje["fitness"],
                    aprendizaje["timestamp"]
                )
        except Exception as e:
            self.logger.error(f"[Aprendizaje] Error publicando aprendizaje: {e}")

    async def consumir_aprendizajes(self):
        """Consume aprendizajes de otros núcleos y los aplica si son mejores."""
        try:
            if not self.redis_client:
                self.logger.warning("[Aprendizaje] Redis no inicializado")
                return
            msgs = await self.redis_client.xread({"corec:aprendizajes": "$"}, block=1000, count=10)
            for _, batch in msgs:
                for _, msg in batch:
                    if msg["instancia_id"] == self.config["instance_id"]:
                        continue  # Ignorar propios aprendizajes
                    for bloque in self.bloques:
                        if msg["bloque_id"] == bloque.id and msg["fitness"] > bloque.fitness:
                            bloque.quantization_step = msg["estrategia"]["quantization_step"]
                            bloque.max_concurrent_tasks = msg["estrategia"]["max_concurrent_tasks"]
                            bloque.increment_factor = msg["estrategia"]["increment_factor"]
                            self.logger.info(f"[Aprendizaje] Aplicado aprendizaje de {msg['instancia_id']} a {bloque.id}")
        except Exception as e:
            self.logger.error(f"[Aprendizaje] Error consumiendo aprendizajes: {e}")

    async def evaluar_estrategias(self):
        """Evalúa estrategias de todos los bloques."""
        try:
            for bloque in self.bloques:
                await self.modules["evolucion"].evaluar_estrategia(bloque)
            self.logger.debug("[Núcleo] Estrategias evaluadas")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error evaluando estrategias: {e}")
            await self.publicar_alerta({
                "tipo": "error_evolucion",
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def ejecutar(self):
        """Mantiene vivo el núcleo; el scheduler gestiona las tareas."""
        try:
            self.logger.info("[Núcleo] Ejecutando ciclo principal (scheduler)...")
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.logger.info("[Núcleo] Ejecución cancelada.")
            raise
        except Exception as e:
            self.logger.error(f"[Núcleo] Error en ejecución continua: {e}")
            await self.publicar_alerta({
                "tipo": "error_ejecucion",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def detener(self):
        """Apaga scheduler, módulos y cierra conexiones."""
        try:
            if self.scheduler:
                self.scheduler.shutdown()
                self.logger.info("[Núcleo] Scheduler detenido")
            for module in self.modules.values():
                await module.detener()
            if self.redis_client:
                await self.redis_client.close()
            if self.db_pool:
                self.db_pool.close()
            self.logger.info("[Núcleo] Detención completa")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error durante la detención: {e}")
            await self.publicar_alerta({
                "tipo": "error_detencion",
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def ejecutar_plugin(self, plugin_id: str, comando: dict):
        """Ejecuta un comando en un plugin."""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} no encontrado")
        return await plugin.manejar_comando(comando)
