import asyncio
import json
import time
from pathlib import Path
import asyncpg
import redis.asyncio as aioredis
import pandas as pd
from corec.config_loader import load_config, CoreCConfig
from corec.scheduler import Scheduler
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from corec.modules.ia import ModuloIA
from corec.modules.analisis_datos import ModuloAnalisisDatos
from corec.modules.evolucion import ModuloEvolucion
from corec.modules.ml import ModuloML
from corec.modules.autosanacion import ModuloAutosanacion
from corec.modules.cognitivo import ModuloCognitivo
from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad
from corec.entities_superpuestas import EntidadSuperpuesta
from corec.entrelazador import Entrelazador
from corec.utils.db_utils import init_postgresql, init_redis
from corec.utils.logging import setup_logging


class CoreCNucleus:
    def __init__(self, config_path: str = "config/corec_config.json"):
        self.logger = setup_logging({"log_level": "INFO", "log_file": "corec.log"})
        self.config_path = str(Path(config_path).resolve())
        self.config: CoreCConfig = None
        self.db_pool: asyncpg.Pool = None
        self.redis_client: aioredis.Redis = None
        self.modules = {}
        self.plugins = {}
        self.bloques = []
        self.scheduler = None
        self.entrelazador = None
        self.fallback_storage = Path("fallback_messages.json")
        self.global_concurrent_tasks = 0
        self.global_concurrent_tasks_max = 1000  # Límite global de tareas concurrentes


    async def inicializar(self):
        """Inicializa el núcleo de CoreC, configurando módulos, bloques y conexiones."""
        try:
            self.config = load_config(self.config_path)
            self.db_pool = await init_postgresql(self.config.db_config.model_dump())
            self.redis_client = await init_redis(self.config.redis_config.model_dump())
            self.entrelazador = Entrelazador(self.redis_client, self)
            self.logger.info("Entrelazador inicializado")

            self.modules["registro"] = ModuloRegistro()
            self.modules["sincronizacion"] = ModuloSincronizacion()
            self.modules["ejecucion"] = ModuloEjecucion()
            self.modules["auditoria"] = ModuloAuditoria()
            self.modules["ia"] = ModuloIA()
            self.modules["analisis_datos"] = ModuloAnalisisDatos()
            self.modules["evolucion"] = ModuloEvolucion()
            self.modules["ml"] = ModuloML()
            self.modules["autosanacion"] = ModuloAutosanacion()
            self.modules["cognitivo"] = ModuloCognitivo()

            for name, module in self.modules.items():
                module_config = getattr(self.config, f"{name}_config", {}).model_dump()
                if name == "ia" and not self.config.ia_config.enabled:
                    self.logger.info(f"Módulo {name} deshabilitado")
                    continue
                try:
                    self.logger.debug(f"Inicializando módulo {name}")
                    await module.inicializar(self, module_config)
                except Exception as e:
                    self.logger.error(f"Error inicializando módulo {name}: {e}")
                    await self.publicar_alerta({
                        "tipo": f"error_inicializacion_{name}",
                        "mensaje": str(e),
                        "timestamp": time.time()
                    })

            for block_conf in self.config.bloques:
                entidades = []
                if block_conf.id == "ia_analisis" and self.config.ia_config.enabled:
                    async def ia_fn(carga, mod_ia=self.modules["ia"], bconf=block_conf):
                        datos = await self.get_datos_from_redis(bconf.id)
                        res = await mod_ia.procesar_bloque(None, datos)
                        return res["mensajes"][0] if res["mensajes"] else {"valor": 0.0}
                    entidades = [
                        Entidad(f"ent_{i}", block_conf.canal, ia_fn, block_conf.quantization_step)
                        for i in range(block_conf.entidades)
                    ]
                else:
                    if self.db_pool:
                        async with self.db_pool.acquire() as conn:
                            rows = await conn.fetch(
                                "SELECT entidad_id, roles, quantization_step, min_fitness, mutation_rate "
                                "FROM entidades WHERE bloque_id = $1",
                                block_conf.id
                            )
                            for row in rows:
                                entidades.append(
                                    EntidadSuperpuesta(
                                        row["entidad_id"],
                                        row["roles"],
                                        row["quantization_step"],
                                        row["min_fitness"],
                                        row["mutation_rate"],
                                        nucleus=self
                                    )
                                )
                    if not entidades:
                        entidades = [
                            EntidadSuperpuesta(
                                f"ent_{i}",
                                {"rol1": 0.5, "rol2": 0.5},
                                block_conf.quantization_step,
                                block_conf.autoreparacion.min_fitness,
                                block_conf.mutacion.mutation_rate,
                                nucleus=self
                            )
                            for i in range(block_conf.entidades)
                        ]
                bloque = BloqueSimbiotico(
                    block_conf.id,
                    block_conf.canal,
                    entidades,
                    block_conf.max_size_mb,
                    self,
                    block_conf.quantization_step,
                    block_conf.autoreparacion.max_errores,
                    block_conf.max_concurrent_tasks,
                    block_conf.cpu_intensive,
                    block_conf.mutacion.model_dump(),
                    block_conf.autorreplicacion.model_dump()
                )
                self.bloques.append(bloque)
                for entidad in entidades:
                    self.entrelazador.registrar_entidad(entidad)
                await self.modules["registro"].registrar_bloque(
                    bloque.id, bloque.canal, len(entidades), bloque.max_size_mb
                )

            for bloque in self.bloques:
                entidades = bloque.entidades
                for i in range(0, len(entidades), 2):
                    if i + 1 < len(entidades):
                        try:
                            self.entrelazador.enlazar(entidades[i], entidades[i + 1])
                        except ValueError as e:
                            self.logger.warning(f"Error enlazando entidades en {bloque.id}: {e}")

            self.scheduler = Scheduler(self)
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
            self.scheduler.schedule_periodic(
                func=self.modules["autosanacion"].verificar_estado,
                seconds=120,
                job_id="autosanacion_periodica"
            )
            self.scheduler.schedule_periodic(
                func=self.modules["cognitivo"].generar_metadialogo,
                seconds=300,
                job_id="cognitivo_metadialogo"
            )
            self.scheduler.schedule_periodic(
                func=self.modules["cognitivo"].detectar_contradicciones,
                seconds=600,
                job_id="cognitivo_contradicciones"
            )
            self.scheduler.schedule_periodic(
                func=self.modules["cognitivo"].actualizar_atencion,
                seconds=300,
                job_id="cognitivo_atencion"
            )
            self.scheduler.schedule_periodic(
                func=self.modules["cognitivo"].resolver_conflictos,
                seconds=600,
                job_id="cognitivo_conflictos"
            )

            self.logger.info("Inicialización completa")

        except FileNotFoundError as e:
            self.logger.error(f"Configuración no encontrada: {e}")
            raise
        except ValueError as e:
            self.logger.error(f"Configuración inválida: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error inicializando: {e}")
            await self.publicar_alerta({
                "tipo": "error_inicializacion",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise


    async def consultar_intuicion(self, tipo: str) -> float:
        """Consulta la intuición del módulo cognitivo para un tipo específico."""
        if "cognitivo" not in self.modules:
            raise ValueError("Módulo Cognitivo no inicializado")
        return await self.modules["cognitivo"].intuir(tipo)


    async def procesar_entrelazador(self):
        """Propaga cambios a través del entrelazador."""
        try:
            for bloque in self.bloques:
                cambio = {"fitness": bloque.fitness}
                for entidad in bloque.entidades:
                    await self.entrelazador.afectar(entidad, cambio, max_saltos=1)
            self.logger.debug("Cambios propagados en Entrelazador")
        except Exception as e:
            self.logger.error(f"Error procesando Entrelazador: {e}")
            await self.publicar_alerta({
                "tipo": "error_entrelazador",
                "mensaje": str(e),
                "timestamp": time.time()
            })


    async def process_bloque(self, bloque: BloqueSimbiotico):
        """Procesa un bloque simbiótico."""
        try:
            if self.global_concurrent_tasks >= self.global_concurrent_tasks_max:
                self.logger.warning(
                    f"Límite global de tareas alcanzado: {self.global_concurrent_tasks}"
                )
                await self.publicar_alerta({
                    "tipo": "limite_tareas_global",
                    "bloque_id": bloque.id,
                    "mensaje": f"Límite global de tareas concurrentes alcanzado: {self.global_concurrent_tasks}",
                    "timestamp": time.time()
                })
                return
            self.global_concurrent_tasks += bloque.current_concurrent_tasks
            if bloque.id != "ia_analisis":
                await self.modules["ejecucion"].encolar_bloque(bloque)
            if self.db_pool:
                async with self.db_pool.acquire() as conn:
                    await bloque.escribir_postgresql(conn)
            else:
                self.logger.warning(f"PostgreSQL no disponible, usando fallback para {bloque.id}")
                await self.save_fallback_messages(bloque.id, bloque.mensajes)
                await self.publicar_alerta({
                    "tipo": "error_db_pool",
                    "bloque_id": bloque.id,
                    "mensaje": "db_pool no inicializado, usando fallback",
                    "timestamp": time.time()
                })
            ml_module = self.modules.get("ml")
            if ml_module:
                for entidad in bloque.entidades:
                    if isinstance(entidad, EntidadSuperpuesta):
                        await ml_module.entrenar_modelo(entidad, bloque.fitness)
        except Exception as e:
            self.logger.error(f"Error procesando bloque {bloque.id}: {e}")
            await self.publicar_alerta({
                "tipo": "error_procesamiento_bloque",
                "bloque_id": bloque.id,
                "mensaje": str(e),
                "timestamp": time.time()
            })
        finally:
            self.global_concurrent_tasks = max(0, self.global_concurrent_tasks - bloque.current_concurrent_tasks)


    async def save_fallback_messages(self, bloque_id: str, mensajes: list):
        """Guarda mensajes en un archivo de respaldo si PostgreSQL no está disponible."""
        try:
            if self.fallback_storage.exists():
                with open(self.fallback_storage, "r") as f:
                    existing = json.load(f)
                    if not isinstance(existing, list):
                        self.logger.error(
                            f"Formato inválido en {self.fallback_storage}, esperado lista"
                        )
                        existing = []
            else:
                existing = []
            existing.extend([{"bloque_id": bloque_id, "mensaje": m, "retry_count": 0} for m in mensajes])
            with open(self.fallback_storage, "w") as f:
                json.dump(existing, f)
            self.logger.info(f"Mensajes de {bloque_id} guardados en fallback")
        except Exception as e:
            self.logger.error(f"Error guardando mensajes en fallback: {e}")


    async def retry_fallback_messages(self):
        """Reintenta escribir mensajes de respaldo en PostgreSQL."""
        if not self.db_pool or not self.fallback_storage.exists():
            self.logger.info("No hay mensajes de fallback o db_pool no disponible")
            return
        try:
            self.logger.info(f"Leyendo mensajes de {self.fallback_storage}")
            with open(self.fallback_storage, "r") as f:
                messages = json.load(f)
            if not isinstance(messages, list):
                self.logger.error(f"Formato inválido en {self.fallback_storage}, esperado lista")
                return
            self.logger.info(f"Encontrados {len(messages)} mensajes para reintentar")
            remaining_messages = []
            async with self.db_pool.acquire() as conn:
                self.logger.debug("Adquirida conexión al pool de PostgreSQL")
                for msg in messages:
                    bloque_id = msg.get("bloque_id")
                    m = msg.get("mensaje")
                    retry_count = msg.get("retry_count", 0)
                    if retry_count >= 5:
                        self.logger.warning(f"Mensaje descartado tras 5 intentos: {msg}")
                        continue
                    if not (bloque_id and m):
                        self.logger.warning(f"Mensaje inválido: {msg}")
                        continue
                    self.logger.debug(f"Insertando mensaje para bloque {bloque_id}, entidad {m['entidad_id']}")
                    try:
                        await conn.execute(
                            """
                            INSERT INTO mensajes (
                                bloque_id, entidad_id, canal, valor, clasificacion, probabilidad, timestamp, roles
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """,
                            bloque_id,
                            m["entidad_id"],
                            m["canal"],
                            m["valor"],
                            m.get("clasificacion", ""),
                            m.get("probabilidad", 0.0),
                            m["timestamp"],
                            json.dumps(m.get("roles", {}))
                        )
                    except Exception as e:
                        self.logger.error(f"Error insertando mensaje {msg}: {e}")
                        msg["retry_count"] = retry_count + 1
                        remaining_messages.append(msg)
            if remaining_messages:
                with open(self.fallback_storage, "w") as f:
                    json.dump(remaining_messages, f)
                self.logger.info(f"{len(remaining_messages)} mensajes permanecen en fallback")
            else:
                self.fallback_storage.unlink(missing_ok=True)
                self.logger.info("Mensajes de fallback escritos en PostgreSQL y archivo eliminado")
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decodificando JSON en {self.fallback_storage}: {e}")
        except Exception as e:
            self.logger.error(f"Error reintentando mensajes de fallback: {e}")


    async def ejecutar_analisis(self):
        """Ejecuta análisis de datos sobre mensajes recientes."""
        try:
            if not self.db_pool:
                self.logger.warning("No se puede ejecutar análisis, db_pool no inicializado")
                return
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT bloque_id, valor FROM mensajes LIMIT 1000")
            if not rows:
                self.logger.debug("No hay mensajes para análisis")
                return
            df = pd.DataFrame(rows, columns=["bloque_id", "valor"])
            df_wide = df.pivot(columns="bloque_id", values="valor").fillna(0)
            await self.modules["analisis_datos"].analizar(df_wide, "mensajes_recientes")
        except Exception as e:
            self.logger.error(f"Error en ejecutar_analisis: {e}")
            await self.publicar_alerta({
                "tipo": "error_analisis",
                "mensaje": str(e),
                "timestamp": time.time()
            })


    async def get_datos_from_redis(self, bloque_id: str) -> dict:
        """Obtiene datos de Redis para un bloque específico."""
        try:
            if not self.redis_client:
                self.logger.warning(f"Redis no inicializado para {bloque_id}")
                return {"valores": []}
            max_length = self.config.redis_config.stream_max_length
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
            self.logger.error(f"Error leyendo Redis para {bloque_id}: {e}")
            await self.publicar_alerta({
                "tipo": "error_redis_datos",
                "bloque_id": bloque_id,
                "mensaje": str(e),
                "timestamp": time.time()
            })
            return {"valores": []}


    async def publicar_alerta(self, alerta: dict):
        """Publica una alerta en Redis o la archiva si Redis no está disponible."""
        try:
            if not self.redis_client:
                self.logger.warning("Redis no inicializado, archivando localmente")
                await self.archive_alert(alerta)
                return
            key = f"alertas:{alerta['tipo']}"
            max_length = self.config.redis_config.stream_max_length
            await self.redis_client.xadd(key, alerta, maxlen=max_length)
            self.logger.debug(f"Alerta publicada: {alerta}")
            stream_len = await self.redis_client.xlen(key)
            if stream_len >= max_length * 0.9:
                await self.archive_alert(alerta)
        except aioredis.RedisError as e:
            self.logger.error(f"Error publicando alerta: {e}")
            await self.archive_alert(alerta)


    async def archive_alert(self, alerta: dict):
        """Archiva una alerta en PostgreSQL."""
        if not self.db_pool:
            self.logger.warning("PostgreSQL no disponible, no se puede archivar")
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
            self.logger.info(f"Alerta archivada en PostgreSQL: {alerta['tipo']}")
        except Exception as e:
            self.logger.error(f"Error archivando alerta: {e}")


    async def publicar_aprendizaje(self, aprendizaje: dict):
        """Publica un aprendizaje en Redis y PostgreSQL."""
        try:
            if not self.redis_client:
                self.logger.warning("Redis no inicializado")
                return
            aprendizaje["instancia_id"] = self.config.instance_id
            aprendizaje["timestamp"] = time.time()
            await self.redis_client.xadd("corec:aprendizajes", aprendizaje, maxlen=1000)
            self.logger.info(f"Aprendizaje publicado: {aprendizaje}")
            if self.db_pool:
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
            self.logger.error(f"Error publicando aprendizaje: {e}")


    async def consumir_aprendizajes(self):
        """Consume aprendizajes de otras instancias y los aplica localmente."""
        try:
            if not self.redis_client:
                self.logger.warning("Redis no inicializado")
                return
            msgs = await self.redis_client.xread({"corec:aprendizajes": "$"}, block=1000, count=10)
            for _, batch in msgs:
                for _, msg in batch:
                    if msg["instancia_id"] == self.config.instance_id:
                        continue
                    for bloque in self.bloques:
                        if msg["bloque_id"] == bloque.id and msg["fitness"] > bloque.fitness:
                            bloque.quantization_step = msg["estrategia"]["quantization_step"]
                            bloque.max_concurrent_tasks = msg["estrategia"]["max_concurrent_tasks"]
                            bloque.increment_factor = msg["estrategia"]["increment_factor"]
                            self.logger.info(f"Aplicado aprendizaje de {msg['instancia_id']} a {bloque.id}")
        except Exception as e:
            self.logger.error(f"Error consumiendo aprendizajes: {e}")


    async def evaluar_estrategias(self):
        """Evalúa estrategias para optimizar el rendimiento de los bloques."""
        try:
            for bloque in self.bloques:
                await self.modules["evolucion"].evaluar_estrategia(bloque)
            self.logger.debug("Estrategias evaluadas")
        except Exception as e:
            self.logger.error(f"Error evaluando estrategias: {e}")
            await self.publicar_alerta({
                "tipo": "error_evolucion",
                "mensaje": str(e),
                "timestamp": time.time()
            })


    async def ejecutar(self):
        """Ejecuta el ciclo principal del núcleo."""
        try:
            self.logger.info("Ejecutando ciclo principal (scheduler)...")
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.logger.info("Ejecución cancelada.")
            raise
        except Exception as e:
            self.logger.error(f"Error en ejecución continua: {e}")
            await self.publicar_alerta({
                "tipo": "error_ejecucion",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise


    async def detener(self):
        """Detiene el núcleo y sus componentes."""
        try:
            if self.scheduler:
                self.scheduler.shutdown()
                self.logger.info("Scheduler detenido")
            for module in self.modules.values():
                await module.detener()
            if self.redis_client:
                await self.redis_client.close()
            if self.db_pool:
                await self.db_pool.close()
            self.logger.info("Detención completa")
        except Exception as e:
            self.logger.error(f"Error durante la detención: {e}")
            await self.publicar_alerta({
                "tipo": "error_detencion",
                "mensaje": str(e),
                "timestamp": time.time()
            })


    async def ejecutar_plugin(self, plugin_id: str, comando: dict):
        """Ejecuta un comando en un plugin específico."""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin {plugin_id} no encontrado")
        return await plugin.manejar_comando(comando)
