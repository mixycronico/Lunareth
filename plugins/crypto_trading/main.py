import asyncio
import logging
import yaml
from typing import Dict, Any
from corec.core import ComponenteBase
from plugins.crypto_trading.processors.orchestrator_processor import OrchestratorProcessor

class CryptoTrading(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("CryptoTrading")
        self.nucleus = None
        self.redis_client = None
        self.config = None
        self.orchestrator = None

    async def inicializar(self, nucleus, config=None):
        try:
            self.nucleus = nucleus
            self.redis_client = self.nucleus.redis_client
            if not self.redis_client:
                raise ValueError("Redis client no inicializado")

            self.config = config

            # Crear e inicializar el OrchestratorProcessor
            self.orchestrator = OrchestratorProcessor(self.config, self.nucleus, self.redis_client)
            await self.orchestrator.initialize()

            self.logger.info("[CryptoTrading] Plugin inicializado correctamente")
        except Exception as e:
            self.logger.error(f"[CryptoTrading] Error al inicializar: {e}")
            await self.nucleus.publicar_alerta({
                "tipo": "error_inicializacion_plugin",
                "plugin_id": "crypto_trading",
                "mensaje": str(e),
                "timestamp": datetime.datetime.utcnow().timestamp()
            })
            raise

    async def manejar_comando(self, comando: Dict[str, Any]) -> Dict[str, Any]:
        try:
            action = comando.get("action")
            params = comando.get("params", {})

            if action == "ejecutar_operacion":
                return await self.orchestrator.execution_processor.ejecutar_operacion(params.get("exchange"), {
                    "precio": params.get("precio", 50000),
                    "cantidad": params.get("cantidad", 0.1),
                    "activo": params.get("pair"),
                    "tipo": params.get("side")
                }, paper_mode=self.orchestrator.paper_mode, trade_multiplier=1).__next__()
            else:
                return {"status": "error", "message": f"Acción no soportada: {action}"}
        except Exception as e:
            self.logger.error(f"[CryptoTrading] Error al manejar comando: {e}")
            return {"status": "error", "message": str(e)}

    async def detener(self):
        await self.orchestrator.detener()
        self.logger.info("[CryptoTrading] Plugin detenido")
