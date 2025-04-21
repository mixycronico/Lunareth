import asyncio
import logging
import datetime
from typing import Dict, Any
from corec.core import ComponenteBase
from plugins.crypto_trading.processors.orchestrator_processor import OrchestratorProcessor
from plugins.crypto_trading.config_loader import load_config_dict

class CryptoTrading(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("CryptoTrading")
        self.nucleus = None
        self.redis_client = None
        self.config = None
        self.orchestrator = None
        self.plugin_loaded = False

    async def inicializar(self, nucleus, config=None):
        try:
            self.nucleus = nucleus
            self.redis_client = self.nucleus.redis_client
            if not self.redis_client:
                raise ValueError("Redis client no inicializado")

            # Cargar y validar la configuración con config_loader
            # config.json se usa solo para la comunicación con CoreC
            # La configuración interna del plugin se carga desde un archivo separado
            self.config = load_config_dict("config.json")

            # Crear e inicializar el OrchestratorProcessor
            self.orchestrator = OrchestratorProcessor(self.config, self.nucleus, self.redis_client)
            await self.orchestrator.initialize()
            self.plugin_loaded = True

            self.logger.info("[CryptoTrading] Plugin inicializado correctamente")
        except Exception as e:
            self.logger.error(f"[CryptoTrading] Error al inicializar: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion_plugin",
                "plugin_id": "crypto_trading",
                "mensaje": str(e),
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            raise

    async def load_plugin(self):
        """Carga dinámicamente el plugin si no está cargado."""
        if self.plugin_loaded:
            self.logger.info("[CryptoTrading] Plugin ya está cargado")
            return
        try:
            await self.inicializar(self.nucleus, self.config)
            self.logger.info("[CryptoTrading] Plugin cargado dinámicamente")
        except Exception as e:
            self.logger.error(f"[CryptoTrading] Error al cargar plugin: {e}")
            raise

    async def unload_plugin(self):
        """Descarga dinámicamente el plugin si está cargado."""
        if not self.plugin_loaded:
            self.logger.info("[CryptoTrading] Plugin no está cargado")
            return
        try:
            await self.orchestrator.detener()
            self.plugin_loaded = False
            self.logger.info("[CryptoTrading] Plugin descargado dinámicamente")
        except Exception as e:
            self.logger.error(f"[CryptoTrading] Error al descargar plugin: {e}")
            raise

    async def manejar_comando(self, comando: Dict[str, Any]) -> Dict[str, Any]:
        try:
            action = comando.get("action")
            params = comando.get("params", {})

            if action == "ejecutar_operacion":
                if not self.plugin_loaded:
                    return {"status": "error", "message": "Plugin no está cargado"}
                return await self.orchestrator.execution_processor.ejecutar_operacion(params.get("exchange"), {
                    "precio": params.get("precio", 50000),
                    "cantidad": params.get("cantidad", 0.1),
                    "activo": params.get("pair"),
                    "tipo": params.get("side")
                }, paper_mode=self.orchestrator.paper_mode, trade_multiplier=1).__next__()
            elif action == "load_plugin":
                await self.load_plugin()
                return {"status": "success", "message": "Plugin cargado"}
            elif action == "unload_plugin":
                await self.unload_plugin()
                return {"status": "success", "message": "Plugin descargado"}
            else:
                return {"status": "error", "message": f"Acción no soportada: {action}"}
        except Exception as e:
            self.logger.error(f"[CryptoTrading] Error al manejar comando: {e}")
            return {"status": "error", "message": str(e)}

    async def detener(self):
        if self.plugin_loaded:
            await self.unload_plugin()
        self.logger.info("[CryptoTrading] Plugin detenido")
