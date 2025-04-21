import asyncio
import logging
import yaml
from typing import Dict, Any
from corec.core import ComponenteBase
from corec.entities import crear_entidad
from plugins.crypto_trading.blocks.trading_block import TradingBlock
from plugins.crypto_trading.blocks.monitor_block import MonitorBlock
from plugins.crypto_trading.processors.analyzer_processor import AnalyzerProcessor
from plugins.crypto_trading.processors.capital_processor import CapitalProcessor
from plugins.crypto_trading.processors.exchange_processor import ExchangeProcessor
from plugins.crypto_trading.processors.execution_processor import ExecutionProcessor
from plugins.crypto_trading.processors.macro_processor import MacroProcessor
from plugins.crypto_trading.processors.monitor_processor import MonitorProcessor
from plugins.crypto_trading.processors.predictor_processor import PredictorProcessor
from plugins.crypto_trading.strategies.momentum_strategy import MomentumStrategy
from plugins.crypto_trading.utils.db import TradingDB
import datetime

class CryptoTrading(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("CryptoTrading")
        self.nucleus = None
        self.redis_client = None
        self.trading_pairs = ["ALT3/USDT", "ALT4/USDT"]
        self.trading_blocks = []
        self.monitor_blocks = []
        self.analyzer_processor = None
        self.capital_processor = None
        self.exchange_processor = None
        self.execution_processor = None
        self.macro_processor = None
        self.monitor_processor = None
        self.predictor_processor = None
        self.strategy = MomentumStrategy()
        self.trading_db = None
        self.paper_mode = True  # Modo paper activado por defecto para pruebas
        self.open_trades = {}  # Seguimiento de operaciones abiertas

    async def inicializar(self, nucleus, config=None):
        """Inicializa el plugin CryptoTrading."""
        try:
            self.nucleus = nucleus
            self.redis_client = self.nucleus.redis_client
            if not self.redis_client:
                raise ValueError("Redis client no inicializado")

            # Configuración del modo paper desde config
            self.paper_mode = config.get("paper_mode", True)
            self.logger.info(f"[CryptoTrading] Modo paper: {self.paper_mode}")

            # Inicializar base de datos independiente para CryptoTrading
            db_config = {
                "dbname": "crypto_trading_db",
                "user": "postgres",
                "password": "secure_password",
                "host": "localhost",
                "port": 5432
            }
            self.trading_db = TradingDB(db_config)
            await self.trading_db.connect()

            # Inicializar procesadores
            self.analyzer_processor = AnalyzerProcessor(config, self.redis_client)
            self.capital_processor = CapitalProcessor(config)
            self.exchange_processor = ExchangeProcessor(config, self.nucleus)
            self.execution_processor = ExecutionProcessor(config, self.redis_client)
            self.macro_processor = MacroProcessor(config, self.redis_client)
            self.monitor_processor = MonitorProcessor(config, self.redis_client)
            self.predictor_processor = PredictorProcessor(config, self.redis_client)
            await self.predictor_processor.inicializar()

            # Inicializar bloques simbióticos
            trading_entities = [
                crear_entidad(f"trade_ent_{i}", 3, lambda carga: {"valor": 0.5})
                for i in range(1000)
            ]
            trading_block = TradingBlock(
                id="trading_block_1",
                canal=3,
                entidades=trading_entities,
                max_size_mb=5,
                nucleus=self.nucleus,
                execution_processor=self.execution_processor,
                trading_db=self.trading_db
            )
            self.trading_blocks.append(trading_block)

            monitor_entities = [
                crear_entidad(f"monitor_ent_{i}", 3, lambda carga: {"valor": 0.5})
                for i in range(1000)
            ]
            monitor_block = MonitorBlock(
                id="monitor_block_1",
                canal=3,
                entidades=monitor_entities,
                max_size_mb=5,
                nucleus=self.nucleus,
                analyzer_processor=self.analyzer_processor,
                monitor_processor=self.monitor_processor
            )
            self.monitor_blocks.append(monitor_block)

            # Registrar bloques compartidos en CoreCNucleus
            self.nucleus.bloques.extend(self.trading_blocks + self.monitor_blocks)

            # Iniciar bucle de monitoreo
            asyncio.create_task(self._monitor_loop())

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
        """Maneja comandos recibidos, como operaciones de trading."""
        try:
            action = comando.get("action")
            params = comando.get("params", {})

            if action == "ejecutar_operacion":
                return await self._execute_trade(params.get("pair"), params.get("side"))
            else:
                return {"status": "error", "message": f"Acción no soportada: {action}"}
        except Exception as e:
            self.logger.error(f"[CryptoTrading] Error al manejar comando: {e}")
            return {"status": "error", "message": str(e)}

    async def _monitor_loop(self):
        """Bucle de monitoreo que verifica el mercado cada 5 minutos."""
        while True:
            try:
                now = datetime.datetime.now()
                start_time = now.replace(hour=6, minute=0, second=0, microsecond=0)
                end_time = now.replace(hour=22, minute=0, second=0, microsecond=0)
                within_trading_hours = start_time <= now <= end_time

                # Monitorear operaciones abiertas siempre
                await self._monitor_open_trades()

                # Ejecutar nuevas operaciones solo dentro del horario
                if within_trading_hours:
                    for block in self.monitor_blocks:
                        result = await block.procesar(0.5)
                        if result["status"] != "success":
                            self.logger.warning(f"Error en monitoreo: {result['motivo']}")
                            continue

                        # Simulación de datos (en producción, usar fetchers reales)
                        macro_data = {"sp500": 0.02, "nasdaq": 0.01, "dxy": -0.01, "gold": 0.005, "oil": 0.03}
                        crypto_data = {"volume": 1000000, "market_cap": 2000000000}
                        sentiment = self.strategy.calculate_momentum(macro_data, crypto_data)
                        side = self.strategy.decide_trade(sentiment)

                        # Ejecutar operación basada en el sentimiento
                        for pair in self.trading_pairs:
                            trade_result = await self._execute_trade(pair, side)
                            if trade_result["status"] == "success":
                                self.open_trades[pair] = trade_result["result"]

                await asyncio.sleep(300)  # Esperar 5 minutos
            except Exception as e:
                self.logger.error(f"[CryptoTrading] Error en bucle de monitoreo: {e}")
                await asyncio.sleep(300)  # Continuar tras error

    async def _monitor_open_trades(self):
        """Monitorea operaciones abiertas fuera del horario de trading."""
        for pair, trade in list(self.open_trades.items()):
            try:
                # Simulación de monitoreo (en producción, verificar precio actual)
                self.logger.info(f"[CryptoTrading] Monitoreando operación abierta para {pair}: {trade}")
                # Ejemplo: Cerrar operación si cumple condiciones
                if "condición de cierre simulada":
                    await self._close_trade(pair, trade)
            except Exception as e:
                self.logger.error(f"[CryptoTrading] Error al monitorear operación abierta para {pair}: {e}")

    async def _close_trade(self, pair: str, trade: dict):
        """Cierra una operación abierta."""
        self.logger.info(f"[CryptoTrading] Cerrando operación para {pair}")
        # Simulación de cierre (en producción, enviar orden de cierre al exchange)
        trade["status"] = "closed"
        trade["close_timestamp"] = datetime.datetime.utcnow().isoformat()
        await self.trading_db.save_order(
            "binance",
            trade["orden_id"],
            pair,
            "spot",
            "closed",
            datetime.datetime.utcnow().timestamp()
        )
        del self.open_trades[pair]
        await self.nucleus.publicar_alerta({
            "tipo": "operacion_cerrada",
            "plugin_id": "crypto_trading",
            "pair": pair,
            "timestamp": trade["close_timestamp"]
        })

    async def _execute_trade(self, pair: str, side: str) -> Dict[str, Any]:
        """Ejecuta una operación de trading usando el TradingBlock."""
        for block in self.trading_blocks:
            result = await block.procesar(0.5, pair, side, paper_mode=self.paper_mode)
            if result["status"] == "success":
                return result
        return {"status": "error", "motivo": "No se pudo ejecutar la operación"}

    async def detener(self):
        """Detiene el plugin CryptoTrading."""
        await self.exchange_processor.close()
        await self.trading_db.disconnect()
        self.logger.info("[CryptoTrading] Plugin detenido")
