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
from corec.modules.ia import ModuloIA  # Nueva importación
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
                    f"redis://{redis_config.get('username', 'default')}:{redis_config.get('password', 'default')}@{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}"
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
            self.modules["ia"] = ModuloIA()  # Nuevo módulo
            for name, module in self.modules.items():
                await module.inicializar(self, self.config.get(f"{name}_config"))

            # Crear bloques simbióticos
            for block_config in self.config.get("bloques", []):
                try:
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
                    seconds=1,
                    job_id=f"process_{bloque.id}",
                    args=[bloque]
                )

            if len(self.bloques) >= 2:
                self.scheduler.schedule_periodic(
                    func=self.synchronize_bloques,
                    seconds=1,
                    job_id="synchronize_bloques",
                    args=[self.bloques[0], self.bloques[1], 0.1, self.bloques[1].canal]
                )

            self.scheduler.schedule_periodic(
                func=self.modules["auditoria"].detectar_anomalias,
                seconds=1,
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

    async def process_bloque(self, bloque: BloqueSimbiotico):
        """Procesa un bloque simbiótico y escribe los resultados."""
        try:
            # Procesar con IA si es ia_analisis
            if bloque.id == "ia_analisis" and "ia" in self.modules:
                datos = {"valores": [random.uniform(0, 1) for _ in range(10)]}  # Ejemplo, ajusta según datos reales
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

    # Resto del código sin cambios (synchronize_bloques, registrar_plugin, etc.)
    # ...

async def init_postgresql(config: Dict[str, Any]):
    try:
        return await asyncpg.create_pool(**config)
    except Exception as e:
        logging.error(f"Error creando pool de PostgreSQL: {e}")
        raise
