import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Dict, Any

import aioredis
import asyncpg
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from corec.blocks import BloqueSimbiotico
from corec.config_loader import load_config_dict
from corec.entities import crear_entidad
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.auditoria import ModuloAuditoria
from corec.modules.ia import ModuloIA

logger = logging.getLogger("CoreCNucleus")


class CoreCNucleus:
    def __init__(self, config_path: str):
        self.logger = logging.getLogger("CoreCNucleus")
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.db_pool = None
        self.redis_client = None
        self.modules: Dict[str, Any] = {}
        self.plugins: Dict[str, Any] = {}
        self.bloques: list[BloqueSimbiotico] = []
        self.scheduler: AsyncIOScheduler | None = None

    async def inicializar(self):
        """Carga configuración, conecta a PostgreSQL y Redis, inicializa módulos, bloques y scheduler."""
        try:
            # Carga y valida config
            self.config = load_config_dict(self.config_path)

            # PostgreSQL
            try:
                self.db_pool = await asyncpg.create_pool(**self.config["db_config"])
                async with self.db_pool.acquire() as conn:
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
            except Exception as e:
                self.logger.error(f"[Núcleo] Error inicializando PostgreSQL: {e}")
                await self.publicar_alerta({
                    "tipo": "error_conexion_postgresql",
                    "mensaje": str(e),
                    "timestamp": time.time()
                })

            # Redis
            try:
                rc = self.config["redis_config"]
                url = (
                    f"redis://{rc['username']}:{rc['password']}@"
                    f"{rc['host']}:{rc['port']}"
                )
                self.redis_client = aioredis.from_url(url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("[Redis] Cliente inicializado correctamente")
            except Exception as e:
                self.logger.error(f"[Núcleo] Error conectando a Redis: {e}")
                await self.publicar_alerta({
                    "tipo": "error_conexion_redis",
                    "mensaje": str(e),
                    "timestamp": time.time()
                })

            # Inicializar módulos
            self.modules = {
                "registro": ModuloRegistro(),
                "sincronizacion": ModuloSincronizacion(),
                "ejecucion": ModuloEjecucion(),
                "auditoria": ModuloAuditoria(),
                "ia": ModuloIA(),
            }
            for name, mod in self.modules.items():
                await mod.inicializar(self, self.config.get(f"{name}_config"))

            # Crear bloques simbióticos
            for bc in self.config["bloques"]:
                # Generar entidades
                entidades = []
                if bc["id"] == "ia_analisis":
                    # callback que usa el módulo IA
                    async def ia_cb(carga: float, modulo=self.modules["ia"]) -> Dict[str, Any]:
                        datos = await self.get_datos_from_redis(bc["id"])
                        res = await modulo.procesar_batch(None, datos)
                        # devolvemos lista de mensajes, pero aquí sólo uno
                        return res[0] if res else {
                            "valor": 0.0,
                            "clasificacion": "",
                            "probabilidad": 0.0
                        }

                    entidades = [
                        crear_entidad(f"ia_ent_{i}", bc["canal"], ia_cb)
                        for i in range(bc["entidades"])
                    ]
                else:
                    entidades = [
                        crear_entidad(f"ent_{i}", bc["canal"], lambda carga: {"valor": 0.5})
                        for i in range(bc["entidades"])
                    ]

                bloque = BloqueSimbiotico(
                    bc["id"],
                    bc["canal"],
                    entidades,
                    bc["max_size_mb"],
                    self
                )
                self.bloques.append(bloque)
                await self.modules["registro"].registrar_bloque(
                    bc["id"],
                    bc["canal"],
                    bc["entidades"],
                    bc["max_size_mb"]
                )

            # Scheduler
            self.scheduler = AsyncIOScheduler()
            self.scheduler.start()
            # Programar tareas periódicas
            for b in self.bloques:
                self.scheduler.add_job(
                    self.process_bloque,
                    'interval',
                    seconds=5,
                    args=[b],
                    id=f"process_{b.id}",
                    replace_existing=True,
                    coalesce=True,
                    misfire_grace_time=30
                )
            if len(self.bloques) > 1:
                self.scheduler.add_job(
                    self.synchronize_bloques,
                    'interval',
                    seconds=5,
                    args=[self.bloques[0], self.bloques[1], 0.1, self.bloques[1].canal],
                    id="sync_bloques",
                    replace_existing=True,
                    coalesce=True,
                    misfire_grace_time=30
                )
            self.scheduler.add_job(
                self.modules["auditoria"].detectar_anomalias,
                'interval',
                seconds=5,
                id="audit_anomalies",
                replace_existing=True,
                coalesce=True,
                misfire_grace_time=30
            )

            self.logger.info("[Núcleo] Inicialización completa")

        except Exception as e:
            self.logger.error(f"[Núcleo] Error en inicializar: {e}")
            await self.publicar_alerta({
                "tipo": "error_inicializacion",
                "mensaje": str(e),
                "timestamp": time.time()
            })
            raise

    async def get_datos_from_redis(self, bloque_id: str) -> Dict[str, Any]:
        """Lee datos necesarios desde Redis para IA."""
        try:
            msgs = await self.redis_client.xread({"alertas:datos": "$"}, block=500, count=20)
            valores = []
            for _, entries in msgs:
                for _, m in entries:
                    if m.get("bloque_id") in {"enjambre_sensor", "crypto_trading", "nodo_seguridad"}:
                        valores.extend(m.get("valores", []))
            return {"valores": valores}
        except Exception as e:
            self.logger.error(f"[Núcleo] Error leyendo Redis: {e}")
            return {"valores": []}

    async def process_bloque(self, bloque: BloqueSimbiotico):
        """Procesa y persiste un bloque; usa batching para IA."""
        try:
            if bloque.id == "ia_analisis" and "ia" in self.modules:
                # Preparar lista de datos para cada entidad
                datos_list = [
                    await self.get_datos_from_redis(bloque.id)
                    for _ in bloque.entidades
                ]
                # Inferencia batched
                batch_msgs = await self.modules["ia"].procesar_batch(bloque, datos_list)
                bloque.mensajes.extend(batch_msgs)
            else:
                await self.modules["ejecucion"].encolar_bloque(bloque)

            # Persistir en PostgreSQL
            if self.db_pool:
                await bloque.escribir_postgresql(self.db_pool)
            else:
                await self.publicar_alerta({
                    "tipo": "error_db_pool",
                    "bloque_id": bloque.id,
                    "mensaje": "db_pool no inicializado",
                    "timestamp": time.time()
                })

        except Exception as e:
            await self.publicar_alerta({
                "tipo": "error_procesamiento_bloque",
                "bloque_id": bloque.id,
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def synchronize_bloques(
        self,
        bloque_origen: BloqueSimbiotico,
        bloque_destino: BloqueSimbiotico,
        proporcion: float,
        canal: int
    ):
        """Redirige entidades entre dos bloques periódicamente."""
        try:
            await self.modules["sincronizacion"].redirigir_entidades(
                bloque_origen, bloque_destino, proporcion, canal
            )
        except Exception as e:
            await self.publicar_alerta({
                "tipo": "error_sincronizacion",
                "mensaje": str(e),
                "timestamp": time.time()
            })

    async def publicar_alerta(self, alerta: Dict[str, Any]):
        """Publica una alerta en Redis Streams."""
        if not self.redis_client:
            return
        key = f"alertas:{alerta['tipo']}"
        try:
            await self.redis_client.xadd(key, alerta)
        except Exception:
            pass

    async def ejecutar(self):
        """Mantiene el bucle principal vivo."""
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass

    async def detener(self):
        """Detiene scheduler, módulos, y cierra conexiones."""
        try:
            if self.scheduler:
                self.scheduler.shutdown()
                self.logger.info("[Núcleo] Scheduler detenido")
            for mod in self.modules.values():
                await mod.detener()
            if self.redis_client:
                await self.redis_client.close()
            if self.db_pool:
                await self.db_pool.close()
            self.logger.info("[Núcleo] Detención completa")
        except Exception as e:
            self.logger.error(f"[Núcleo] Error deteniendo: {e}")
            await self.publicar_alerta({
                "tipo": "error_detencion",
                "mensaje": str(e),
                "timestamp": time.time()
            })
