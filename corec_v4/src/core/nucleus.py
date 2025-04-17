import json
import asyncio
from typing import Dict, Any, Optional
from ..utils.config import load_secrets, load_config
from ..utils.logging import logger
from ..utils.openrouter import OpenRouterClient
from ..plugins.plugin_manager import PluginManager
from .modules.registro import ModuloRegistro
from .modules.ejecucion import ModuloEjecucion
from .modules.sincronizacion import ModuloSincronizacion
from .modules.auditoria import ModuloAuditoria
from .processors.base import ProcesadorBase
from .processors.default import DefaultProcessor
import redis.asyncio as aioredis
import zstandard as zstd
import psycopg2

class CoreCNucleus:
    def __init__(self, config_path: str = "configs/core/corec_config_corec1.json", instance_id: str = "corec1"):
        self.logger = logger.getLogger(f"CoreCNucleus-{instance_id}")
        self.instance_id = instance_id
        self.config = load_config(config_path)
        self.db_config = load_secrets("configs/core/secrets/db_config.yaml")
        redis_config = load_secrets("configs/core/secrets/redis_config.yaml")
        self.redis_client = aioredis.from_url(f"redis://{redis_config['host']}:{redis_config['port']}")
        self.openrouter = OpenRouterClient()
        self.plugin_manager = PluginManager()
        self.canales_criticos = ["reparadora_acciones", "seguridad_alertas", "coordinacion_nodos"]
        self.modulos = {
            "registro": ModuloRegistro(),
            "ejecucion": ModuloEjecucion(),
            "sincronizacion": ModuloSincronizacion(),
            "auditoria": ModuloAuditoria()
        }
        self.rol = self.config.get("rol", "generica")

    async def inicializar(self):
        await self.openrouter.initialize()
        await self.plugin_manager.cargar_plugins(self)
        for modulo in self.modulos.values():
            await modulo.inicializar(self)
        self.logger.info(f"[CoreCNucleus-{self.instance_id}] Inicializado")

    def get_procesador(self, canal: str) -> ProcesadorBase:
        processor = self.plugin_manager.get_processor(canal)
        if processor:
            processor.set_nucleus(self)
            return processor
        return DefaultProcessor()

    async def razonar(self, datos: Any, contexto: str) -> Dict[str, Any]:
        return await self.openrouter.analyze(datos, contexto)

    async def responder_chat(self, mensaje: str, contexto: Optional[str] = None) -> Dict[str, Any]:
        return await self.openrouter.chat(mensaje, contexto)

    async def registrar_celu_entidad(self, celu: 'CeluEntidadCoreC'):
        await self.modulo_registro.registrar_celu_entidad(celu)

    async def registrar_micro_celu_entidad(self, micro: 'MicroCeluEntidadCoreC'):
        micro.nucleus = self
        await self.modulo_registro.registrar_micro_celu_entidad(micro)

    async def publicar_alerta(self, alerta: Dict[str, Any]):
        datos_comprimidos = zstd.compress(json.dumps(alerta).encode())
        try:
            conn = psycopg2.connect(**self.db_config)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO eventos (canal, datos, timestamp, instance_id) VALUES (%s, %s, %s, %s)",
                ("alertas", datos_comprimidos, time.time(), self.instance_id)
            )
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            self.logger.error(f"[CoreCNucleus] Error publicando alerta: {e}")

    async def iniciar(self):
        await self.inicializar()
        tasks = [modulo.ejecutar() for modulo in self.modulos.values()]
        await asyncio.gather(*tasks)

    async def detener(self):
        await self.plugin_manager.detener_plugins()
        for modulo in self.modulos.values():
            await modulo.detener()
        await self.openrouter.close()
        await self.redis_client.close()
        self.logger.info(f"[CoreCNucleus-{self.instance_id}] Detenido")

    @property
    def modulo_registro(self):
        return self.modulos["registro"]