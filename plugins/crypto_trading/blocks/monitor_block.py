from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad
import logging
import json

class MonitorBlock(BloqueSimbiotico):
    def __init__(self, id: str, canal: int, entidades: list[Entidad], max_size_mb: float, nucleus, analyzer_processor, monitor_processor):
        super().__init__(id, canal, entidades, max_size_mb, nucleus)
        self.logger = logging.getLogger("MonitorBlock")
        self.analyzer_processor = analyzer_processor
        self.monitor_processor = monitor_processor

    async def procesar(self, carga: float) -> dict:
        """Procesa monitoreo de pares de criptomonedas y calcula tendencias."""
        try:
            result = await super().procesar(carga)
            # Obtener datos de Redis (almacenados por AnalyzerProcessor y MonitorProcessor)
            analysis_data = await self.redis.get("analyzer_data")
            volatility_data = await self.redis.get("volatility_data")

            analysis = json.loads(analysis_data) if analysis_data else {"status": "error", "motivo": "No data"}
            volatility = json.loads(volatility_data) if volatility_data else {"status": "error", "motivo": "No data"}

            if analysis["status"] != "ok" or volatility["status"] != "ok":
                return {"status": "error", "motivo": "Error en análisis o volatilidad"}

            combined_result = {
                "tendencias": analysis["data"],
                "volatilidad": volatility["datos"]
            }
            await self.nucleus.publicar_alerta({
                "tipo": "monitoreo_mercado",
                "plugin_id": "crypto_trading",
                "data": combined_result,
                "timestamp": datetime.datetime.utcnow().timestamp()
            })
            self.logger.info(f"[MonitorBlock {self.id}] Monitoreo completado")
            return {"status": "success", "result": combined_result}
        except Exception as e:
            self.logger.error(f"[MonitorBlock {self.id}] Error al procesar monitoreo: {e}")
            return {"status": "error", "motivo": str(e)}
