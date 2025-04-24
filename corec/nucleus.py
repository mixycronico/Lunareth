import logging
import aioredis
import asyncpg
from typing import Dict, Any
from corec.config_loader import load_config_dict
from corec.scheduler import Scheduler
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from corec.modules.ia import ModuloIA
from corec.entities import crear_entidad
from corec.blocks import BloqueSimbiotico
from plugins import PluginBlockConfig, PluginCommand
import random
import asyncio

class CoreCNucleus:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger("CoreCNucleus")
        self.config_path = config_path
        self.config = None
        self.db_pool = None
        self.redis_client = None
        self.modules = {}
        self.plugins = {}
        self.bloques = []
        self.scheduler = None

    async def inicializar(self):
        try:
            # Cargar configuración
            self.config = load_config_dict(self.config_path)

            # Inicializar conexiones a PostgreSQL y Redis
            try:
                self.db_pool = await init_postgresql(self.config.get("db_config", {}))
            except Exception as e:
                self.logger.error(f"[Núcleo] Error conectando a PostgreSQL: {e}")
                self.db_pool = None
                await self.publicar_alerta({
                    "tipo": "error_conexion_postgresql",
                    "mensaje": str(e),
                    "timestamp": random.random()
                })

            try:
                redis_config = self.config.get("redis_config", {})
                self.redis_client = aioredis.from_url(
                    f"redis://{redis_config.get('username', 'default')}:{redis_config.get('password', 'default')}@{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}",
                    decode_responses=True
                )
                await self.redis_client.ping()
            except Exception as e:
                self.logger.error(f"[Núcleo] Error conectando a Redis: {e}")
                self.redis_client = None
                await self.publicar_alerta({
                    "tipo": "error_conexion_redis",
                    "mensaje": str(e),
                    "timestamp": random.random()
                })

            # Inicializar módulos
            self.modules["registro"] = ModuloRegistro()
            self.modules["sincronizacion"] = ModuloSincronizacion()
            self.modules["ejecucion"] = ModuloEjecucion()
            self.modules["auditoria"] = ModuloAuditoria()
            self.modules["ia"] = ModuloIA()
            for name, module in self.modules.items():
                await module.inicializar(self, self.config.get(f"{name}_config"))

            # Crear bloques simbióticos
            for block_config in self.config.get("bloques", []):
                try:
                    if block_config["id"] == "ia_analisis":
                        # Entidades para ia_analisis usan ModuloIA
                        async def ia_procesar(carga: float, modulo_ia=self.modules["ia"]) -> Dict[str, Any]:
                            datos = await self.get_datos_from_redis(block_config["id"])
                            result = await modulo_ia.procesar_bloque(None, datos)
                            mensaje = result["mensajes"][0] if result["mensajes"] else {
                                "valor": 0.5,
                                "clasificacion": "",
                                "probabilidad": 0.0
                            }
                            return mensaje
                        entidades = [crear_entidad(f"ent_{i}", block_config["canal"], lambda carga, modulo_ia=self.modules["ia"]: ia_procesar(carga, modulo_ia))
                                     for i in range(block_config["entidades"])]
                    else:
                        entidades = [crear_entidad(f"ent_{i}", block_config["canal"], lambda carga: {"valor": 0.5})
                                     for i in range(block_config["entidades"])]
                    bloque = BloqueSimbiotico(
                        block_config["id"],
                        block_config["canal"],
                        entidades,
                        block_config["max_size_mb"],
                        self
                    )
                    self.bloques.append(bloque)
                    await self.modules["registro"].registrar_bloque(
                        block_config["id"],
                        block_config["canal"],
                        block_config["entidades"],
                        block_config["max_size_mb"]
                    )
                except Exception as e:
                    block_id = block_config.get("id", "unknown")
                    await self.publicar_alerta({
                        "tipo": "error_config_bloque",
                        "bloque_id": block_id,
                        "mensaje": str(e),
                        "timestamp": random.random()
                    })

            # Inicializar scheduler
            self.scheduler = Scheduler()
            self.scheduler.start()

            # Programar tareas
            for bloque in self.bloques:
                self.scheduler.schedule_periodic(
                    func=self.process_bloque,
                    seconds=5,  # Ajustado para optimización
                    job_id=f"process_{bloque.id}",
                    args=[bloque]
                )

            if len(self.bloques) >= 2:
                self.scheduler.schedule_periodic(
                    func=self.synchronize_bloques,
                    seconds=5,
                    job_id="synchronize_bloques",
                    args=[self.bloques[0], self.bloques[1], 0.1, self.bloques[1].canal]
                )

            self.scheduler.schedule_periodic(
                func=self.modules["auditoria"].detectar_anomalias,
                seconds=5,
                job_id="audit_anomalies"
            )

            self.logger.info("[Núcleo] Inicialización completa")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error inicializando: {e}")
            await self.publicar_alerta({
                "tipo": "error_inicializacion",
                "mensaje": str(e),
                "timestamp": random.random()
            })
            raise

    async def get_datos_from_redis(self, bloque_id: str) -> Dict[str, Any]:
        """Obtiene datos reales desde Redis para un bloque."""
        try:
            messages = await self.redis_client.xI xread({"alertas:datos": "$"}, block=1000, count=10)
            valores = []
            bloque_origen = "unknown"
            for stream, msgs in messages:
                for msg_id, msg in msgs:
                    if msg.get("bloque_id") in ["enjambre_sensor", "crypto_trading", "nodo_seguridad"]:
                        valores.extend(msg.get("valores", []))
                        bloque_origen = msg.get("bloque_id", "unknown")
            return {"valores": valores, "bloque_origen": bloque_origen}
        except Exception as e:
            self.logger.error(f"[Núcleo] Error obteniendo datos de Redis: {e}")
            await self.publicar_alerta({
                "tipo": "error_redis_datos",
                "bloque_id": bloque_id,
                "mensaje": str(e),
                "timestamp": random.random()
            })
            return {"valores": [], "bloque_origen": "unknown"}

    async def process_bloque(self, bloque: BloqueSimbiotico):
        """Procesa un bloque simbiótico y escribe los resultados en PostgreSQL."""
        try:
            # Procesar con IA si es ia_analisis
            if bloque.id == "ia_analisis" and "ia" in self.modules:
                datos = await self.get_datos_from_redis(bloque.id)
                result = await self.modules["ia"].procesar_bloque(bloque, datos)
                bloque.mensajes.extend(result["mensajes"])
            else:
                await self.modules["ejecucion"].encolar_bloque(bloque)

            if self.db_pool:
                await bloque.escribir_postgresql(self.db_pool)
            else:
                self.logger.warning("[Núcleo] No se puede escribir en PostgreSQL, db_pool no inicializado")
                await self.publicar_alerta({
                    "tipo": "error_db_pool",
                    "bloque_id": bloque.id,
                    "mensaje": "db_pool no inicializado",
                    "timestamp": random.random()
                })
        except Exception as e:
            self.logger.error(f"[Núcleo] Error procesando bloque {bloque.id}: {e}")
            await self.publicar_alerta({
                "tipo": "error_procesamiento_bloque",
                "bloque_id": bloque.id,
                "mensaje": str(e),
                "timestamp": random.random()
            })

    async def synchronize_bloques(self, bloque_origen: BloqueSimbiotico, bloque_destino: BloqueSimbiotico, proporcion: float, canal: int):
        """Sincroniza entidades entre bloques."""
        try:
            await self.modules["sincronizacion"].redirigir_entidades(
                bloque_origen,
                bloque_destino,
                proporcion,
                canal
            )
        except Exception as e:
            self.logger.error(f"[Núcleo] Error en sincronización: {e}")
            await self.publicar_alerta({
                "tipo": "error_sincronizacion",
                "mensaje": str(e),
                "timestamp": random.random()
            })

    def registrar_plugin(self, plugin_id: str, plugin: Any):
        try:
            plugin_config = self.config.get("plugins", {}).get(plugin_id)
            if plugin_config:
                block_config = PluginBlockConfig(**plugin_config["bloque"])
                entidades = [crear_entidad(f"ent_{i}", block_config.canal, lambda carga: {"valor": 0.5})
                             for i in range(block_config.entidades)]
                bloque = BloqueSimbiotico(
                    block_config.bloque_id,
                    block_config.canal,
                    entidades,
                    block_config.max_size_mb,
                    self
                )
                plugin.bloque = bloque
            self.plugins[plugin_id] = plugin
            self.logger.info(f"[Núcleo] Plugin '{plugin_id}' registrado")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error registrando plugin '{plugin_id}': {e}")

    async def ejecutar_plugin(self, plugin_id: str, comando: Dict[str, Any]) -> Dict[str, Any]:
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin '{plugin_id}' no encontrado")
        try:
            comando = PluginCommand(**comando)
            result = await plugin.manejar_comando(comando)
            self.logger.info(f"[Núcleo] Comando ejecutado en plugin '{plugin_id}'")
            return result
        except Exception as e:
            self.logger.error(f"[Núcleo] Error ejecutando comando en plugin '{plugin_id}': {e}")
            return {"status": "error", "message": str(e)}

    async def publicar_alerta(self, alerta: Dict[str, Any]):
        try:
            if not self.redis_client:
                self.logger.warning("[Alerta] Redis client no inicializado, alerta no publicada")
                return
            stream_key = f"alertas:{alerta['tipo']}"
            await self.redis_client.xadd(stream_key, alerta)
            self.logger.warning(f"[Alerta] {alerta}")
        except Exception as e:
            self.logger.error(f"[Alerta] Error publicando alerta: {e}")

    async def ejecutar(self):
        """
        Ejecuta el procesamiento continuo de bloques y plugins.
        Las tareas se programan con APScheduler en inicializar().
        Este método mantiene el sistema en ejecución hasta que se detenga.
        """
        try:
            self.logger.info("[Núcleo] Iniciando ejecución continua...")
            while True:
                await asyncio.sleep(3600)  # Mantener el núcleo en ejecución
        except asyncio.CancelledError:
            self.logger.info("[Núcleo] Ejecución cancelada.")
            raise
        except Exception as e:
            self.logger.error(f"[Núcleo] Error en ejecución continua: {e}")
            await self.publicar_alerta({
                "tipo": "error_ejecucion",
                "mensaje": str(e),
                "timestamp": random.random()
            })
            raise

    async def detener(self):
        try:
            if self.scheduler:
                self.scheduler.shutdown()
                self.logger.info("[Núcleo] Scheduler detenido")
            for module in self.modules.values():
                await module.detener()
            if self.redis_client:
                await self.redis_client.close()
            if self.db_pool:
                await self.db_pool.close()
            self.logger.info("[Núcleo] Detención completa")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error durante la detención: {e}")
            await self.publicar_alerta({
                "tipo": "error_detencion",
                "mensaje": str(e),
                "timestamp": random.random()
            })

async def init_postgresql(config: Dict[str, Any]):
    try:
        pool = await asyncpg.create_pool(**config)
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS bloques (
                    id VARCHAR(50) PRIMARY KEY,
                    canal INTEGER,
                    num_entidades INTEGER,
                    fitness FLOAT,
                    timestamp FLOAT
                );
                CREATE TABLE IF NOT EXISTS mensajes (
                    id SERIAL PRIMARY KEY,
                    bloque_id VARCHAR(50),
                    entidad_id VARCHAR(50),
                    canal INTEGER,
                    valor FLOAT,
                    clasificacion VARCHAR(50),
                    probabilidad FLOAT,
                    timestamp FLOAT
                );
            """)
            await self.publicar_alerta({
                "tipo": "db_inicializada",
                "mensaje": "Tablas bloques y mensajes creadas",
                "timestamp": random.random()
            })
        return pool
    except Exception as e:
        logging.error(f"Error creando pool de PostgreSQL: {e}")
        raise
