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
            # Simulación de datos (en producción, usar fetchers reales)
            precios = {"BTC": [50000, 51000], "ETH": [3000, 3100]}
            volumenes = {"BTC": 1000000, "ETH": 500000}
            
            # Usar AnalyzerProcessor para analizar tendencias
            analysis = await self.analyzer_processor.analizar(precios, volumenes)
            if analysis["status"] != "ok":
                return {"status": "error", "motivo": analysis["motivo"]}
            
            # Usar MonitorProcessor para analizar volatilidad
            volatility = await self.monitor_processor.analizar_volatilidad()
            if volatility["status"] != "ok":
                return {"status": "error", "motivo": volatility["motivo"]}

            # Combinar resultados
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
