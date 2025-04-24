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
from corec.modules.auditoria import ModuloAuditoria
from corec.modules.ejecucion import ModuloEjecucion
from corec.modules.ia import ModuloIA
from corec.modules.registro import ModuloRegistro
from corec.modules.sincronizacion import ModuloSincronizacion
from plugins import PluginBlockConfig, PluginCommand

logger = logging.getLogger("CoreCNucleus")


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
            # 1) Carga y valida config
            self.config = load_config_dict(self.config_path)

            # 2) PostgreSQL
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
                logger.info("[DB] Tablas inicializadas")
            except Exception as e:
                self.logger.error(f"[Núcleo] No se pudo inicializar PostgreSQL: {e}")
                await self.publicar_alerta({
                    "tipo": "error_conexion_postgresql",
                    "mensaje": str(e),
                    "timestamp": random.random()
                })

            # 3) Redis
            try:
                rc = self.config["redis_config"]
                url = f"redis://{rc['username']}:{rc['password']}@{rc['host']}:{rc['port']}"
                self.redis_client = aioredis.from_url(url, decode_responses=True)
                await self.redis_client.ping()
                logger.info("[Redis] Cliente inicializado")
            except Exception as e:
                self.logger.error(f"[Núcleo] No se pudo conectar a Redis: {e}")
                await self.publicar_alerta({
                    "tipo": "error_conexion_redis",
                    "mensaje": str(e),
                    "timestamp": random.random()
                })

            # 4) Módulos
            self.modules = {
                "registro": ModuloRegistro(),
                "sincronizacion": ModuloSincronizacion(),
                "ejecucion": ModuloEjecucion(),
                "auditoria": ModuloAuditoria(),
                "ia": ModuloIA()
            }
            for name, mod in self.modules.items():
                await mod.inicializar(self, self.config.get(f"{name}_config"))

            # 5) Bloques
            for bc in self.config["bloques"]:
                entidades = []
                if bc["id"] == "ia_analisis":
                    # callback que usa el módulo IA
                    async def ia_cb(carga: float, modulo=self.modules["ia"]):
                        datos = await self.get_datos_from_redis(bc["id"])
                        res = await modulo.procesar_bloque(None, datos)
                        # devolvemos primer mensaje o fallback
                        return res["mensajes"][0] if res["mensajes"] else {
                            "valor": 0.0, "clasificacion": "", "probabilidad": 0.0
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

                bloque = BloqueSimbiotico(bc["id"], bc["canal"], entidades, bc["max_size_mb"], self)
                self.bloques.append(bloque)
                await self.modules["registro"].registrar_bloque(
                    bc["id"], bc["canal"], bc["entidades"], bc["max_size_mb"]
                )

            # 6) Scheduler
            self.scheduler = AsyncIOScheduler()
            self.scheduler.start()
            # programa procesamiento y sincronización
            for b in self.bloques:
                self.scheduler.add_job(
                    self.process_bloque, 'interval', seconds=5, args=[b], id=f"proc_{b.id}"
                )
            if len(self.bloques) > 1:
                self.scheduler.add_job(
                    self.synchronize_bloques, 'interval', seconds=5,
                    args=[self.bloques[0], self.bloques[1], 0.1, self.bloques[1].canal],
                    id="sync_blocks"
                )
            self.scheduler.add_job(
                self.modules["auditoria"].detectar_anomalias, 'interval', seconds=5, id="audit"
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
        try:
            msgs = await self.redis_client.xread({"alertas:datos": "$"}, block=500, count=20)
            valores = []
            for _, ms in msgs:
                for _, m in ms:
                    if m.get("bloque_id") in {"enjambre_sensor","crypto_trading","nodo_seguridad"}:
                        valores.extend(m.get("valores", []))
            return {"valores": valores}
        except Exception:
            return {"valores": []}

    async def process_bloque(self, bloque: BloqueSimbiotico):
        try:
            if bloque.id == "ia_analisis":
                # cada entidad hace su ia_cb y acumula mensajes
                for ent in bloque.entidades:
                    msg = await ent.procesar(0.0)
                    bloque.mensajes.append(msg)
            else:
                await self.modules["ejecucion"].encolar_bloque(bloque)

            if self.db_pool:
                await bloque.escribir_postgresql(self.db_pool)
        except Exception as e:
            await self.publicar_alerta({
                "tipo": "error_procesamiento_bloque",
                "bloque_id": bloque.id,
                "mensaje": str(e),
                "timestamp": random.random()
            })

    async def synchronize_bloques(self, src, dst, prop, canal):
        try:
            await self.modules["sincronizacion"].redirigir_entidades(src, dst, prop, canal)
        except Exception as e:
            await self.publicar_alerta({
                "tipo": "error_sincronizacion",
                "mensaje": str(e),
                "timestamp": random.random()
            })

    async def publicar_alerta(self, alerta: Dict[str, Any]):
        if not self.redis_client:
            return
        key = f"alertas:{alerta['tipo']}"
        try:
            await self.redis_client.xadd(key, alerta)
        except:
            pass

    async def ejecutar(self):
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass

    async def detener(self):
        if self.scheduler:
            self.scheduler.shutdown()
        for m in self.modules.values():
            await m.detener()
        if self.redis_client:
            await self.redis_client.close()
        if self.db_pool:
            await self.db_pool.close()
