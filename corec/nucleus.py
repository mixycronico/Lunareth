import logging
import random
import asyncio
import pandas as pd
from typing import Dict, Any, List
from corec.config_loader import load_config_dict
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from corec.modules.ia import ModuloIA
from corec.modules.analisis_datos import ModuloAnalisisDatos
from corec.blocks import BloqueSimbiotico
from corec.entities import crear_entidad
from corec.scheduler import Scheduler
from plugins import PluginBlockConfig, PluginCommand
import asyncpg
import aioredis

async def init_postgresql(config: dict):
    logger = logging.getLogger("CoreCDB")
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
            logger.info("[DB] Tablas 'bloques' y 'mensajes' inicializadas")
        return pool
    except Exception as e:
        logger.error(f"[DB] Error inicializando PostgreSQL: {e}")
        raise

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
            self.config = load_config_dict(self.config_path)
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
                rc = self.config.get("redis_config", {})
                url = f"redis://{rc.get('username')}:{rc.get('password')}@{rc.get('host')}:{rc.get('port')}"
                self.redis_client = aioredis.from_url(url, decode_responses=True)
                await self.redis_client.ping()
            except Exception as e:
                self.logger.error(f"[Núcleo] Error conectando a Redis: {e}")
                self.redis_client = None
                await self.publicar_alerta({
                    "tipo": "error_conexion_redis",
                    "mensaje": str(e),
                    "timestamp": random.random()
                })

            self.modules["registro"] = ModuloRegistro()
            self.modules["sincronizacion"] = ModuloSincronizacion()
            self.modules["ejecucion"] = ModuloEjecucion()
            self.modules["auditoria"] = ModuloAuditoria()
            self.modules["ia"] = ModuloIA()
            self.modules["analisis_datos"] = ModuloAnalisisDatos()

            for name, module in self.modules.items():
                await module.inicializar(self, self.config.get(f"{name}_config"))

            for block_conf in self.config.get("bloques", []):
                entidades = []
                if block_conf["id"] == "ia_analisis":
                    async def ia_fn(carga, mod_ia=self.modules["ia"], bconf=block_conf):
                        datos = await self.get_datos_from_redis(bconf["id"])
                        res = await mod_ia.procesar_bloque(None, datos)
                        return res["mensajes"][0] if res["mensajes"] else {"valor": 0.0}
                    entidades = [
                        crear_entidad(f"ent_{i}", block_conf["canal"], ia_fn)
                        for i in range(block_conf["entidades"])
                    ]
                else:
                    entidades = [
                        crear_entidad(f"ent_{i}", block_conf["canal"], lambda carga: {"valor": 0.5})
                        for i in range(block_conf["entidades"])
                    ]

                bloque = BloqueSimbiotico(
                    block_conf["id"],
                    block_conf["canal"],
                    entidades,
                    block_conf["max_size_mb"],
                    self
                )
                self.bloques.append(bloque)
                await self.modules["registro"].registrar_bloque(
                    bloque.id, bloque.canal, len(entidades), bloque.max_size_mb
                )

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
        try:
            if bloque.id != "ia_analisis":
                await self.modules["ejecucion"].encolar_bloque(bloque)
            if self.db_pool:
                await bloque.escribir_postgresql(self.db_pool)
        except Exception as e:
            self.logger.error(f"[Núcleo] Error procesando bloque {bloque.id}: {e}")
            await self.publicar_alerta({
                "tipo": "error_procesamiento_bloque",
                "bloque_id": bloque.id,
                "mensaje": str(e),
                "timestamp": random.random()
            })

    async def ejecutar_analisis(self):
        try:
            if not self.db_pool:
                return
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch("SELECT bloque_id, valor FROM mensajes")
            if not rows:
                return
            df = pd.DataFrame(rows, columns=["bloque_id", "valor"])
            df_wide = df.pivot(columns="bloque_id", values="valor").fillna(0)
            await self.modules["analisis_datos"].analizar(df_wide, "mensajes_recientes")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error en ejecutar_analisis: {e}")
            await self.publicar_alerta({
                "tipo": "error_analisis",
                "mensaje": str(e),
                "timestamp": random.random()
            })

    async def get_datos_from_redis(self, bloque_id: str) -> dict:
        try:
            msgs = await self.redis_client.xread({"alertas:datos": "$"}, block=1000, count=10)
            valores = []
            for _, batch in msgs:
                for _, msg in batch:
                    if msg.get("bloque_id") in ["enjambre_sensor", "crypto_trading", "nodo_seguridad"]:
                        valores.extend(msg.get("valores", []))
            return {"valores": valores}
        except Exception as e:
            self.logger.error(f"[Núcleo] Error leyendo Redis: {e}")
            await self.publicar_alerta({
                "tipo": "error_redis_datos",
                "bloque_id": bloque_id,
                "mensaje": str(e),
                "timestamp": random.random()
            })
            return {"valores": []}

    async def publicar_alerta(self, alerta: dict):
        try:
            if not self.redis_client:
                self.logger.warning("[Alerta] Redis no inicializado, no se publica alerta")
                return
            key = f"alertas:{alerta['tipo']}"
            await self.redis_client.xadd(key, alerta)
            self.logger.debug(f"[Alerta] {alerta}")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error publicando alerta: {e}")

    async def ejecutar(self):
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
