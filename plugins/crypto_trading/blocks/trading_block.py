from corec.blocks import BloqueSimbiotico
from corec.entities import Entidad
import logging

class TradingBlock(BloqueSimbiotico):
    def __init__(self, id: str, canal: int, entidades: list[Entidad], max_size_mb: float, nucleus, execution_processor, trading_db):
        super().__init__(id, canal, entidades, max_size_mb, nucleus)
        self.logger = logging.getLogger("TradingBlock")
        self.execution_processor = execution_processor
        self.trading_db = trading_db

    async def procesar(self, carga: float, exchange: str, pair: str, side: str, paper_mode: bool = False) -> dict:
        """Procesa una operación de trading específica."""
        try:
            result = await super().procesar(carga)
            # Usar el ExecutionProcessor para realizar la operación
            order = {
                "precio": 50000,  # Simulado, en producción vendría de datos reales
                "cantidad": 0.1,
                "activo": pair,
                "tipo": side
            }
            execution_result = await self.execution_processor.ejecutar_operacion(exchange, order, paper_mode=paper_mode)
            if execution_result["status"] == "ejecutado":
                # Almacenar operación usando la base de datos independiente
                await self.trading_db.save_order(
                    exchange,
                    execution_result["orden_id"],
                    pair,
                    "spot",
                    execution_result["status"],
                    datetime.datetime.utcnow().timestamp()
                )
                await self.nucleus.publicar_alerta({
                    "tipo": "operacion_ejecutada",
                    "plugin_id": "crypto_trading",
                    "exchange": exchange,
                    "pair": pair,
                    "side": side,
                    "amount": execution_result["cantidad"],
                    "price": execution_result["precio"],
                    "timestamp": execution_result["timestamp"],
                    "modo": execution_result["modo"]
                })
                self.logger.info(f"[TradingBlock {self.id}] Operación {side} ejecutada para {exchange}:{pair} ({execution_result['modo']})")
                return {"status": "success", "result": execution_result}
            else:
                self.logger.warning(f"[TradingBlock {self.id}] Operación {side} fallida para {exchange}:{pair}")
                return {"status": "error", "motivo": execution_result["motivo"]}
        except Exception as e:
            self.logger.error(f"[TradingBlock {self.id}] Error al procesar operación: {e}")
            return {"status": "error", "motivo": str(e)}
