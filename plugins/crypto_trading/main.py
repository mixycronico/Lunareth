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
from plugins.crypto_trading.data.alpha_vantage_fetcher import AlphaVantageFetcher
from plugins.crypto_trading.data.coinmarketcap_fetcher import CoinMarketCapFetcher
import datetime

class CryptoTrading(ComponenteBase):
    def __init__(self):
        self.logger = logging.getLogger("CryptoTrading")
        self.nucleus = None
        self.redis_client = None
        self.trading_pairs_by_exchange = {}
        self.trading_blocks = []
        self.monitor_blocks = []
        self.analyzer_processor = None
        self.capital_processor = None
        self.exchange_processor = None
        self.execution_processor = None
        self.macro_processor = None
        self.monitor_processor = None
        self.predictor_processor = None
        self.strategy = None
        self.trading_db = None
        self.paper_mode = True
        self.open_trades = {}
        self.alpha_vantage = None
        self.coinmarketcap = None
        self.config = None
        self.capital = 100  # Capital inicial

    async def inicializar(self, nucleus, config=None):
        """Inicializa el plugin CryptoTrading."""
        try:
            self.nucleus = nucleus
            self.redis_client = self.nucleus.redis_client
            if not self.redis_client:
                raise ValueError("Redis client no inicializado")

            self.config = config
            self.capital = config.get("capital", 100)  # Obtener capital inicial desde config

            self.paper_mode = config.get("paper_mode", True)
            self.logger.info(f"[CryptoTrading] Modo paper: {self.paper_mode}")

            # Inicializar fetchers de datos
            self.alpha_vantage = AlphaVantageFetcher(self.config)
            self.coinmarketcap = CoinMarketCapFetcher(self.config)

            # Inicializar base de datos independiente para CryptoTrading
            db_config = self.config.get("db_config", {
                "dbname": "crypto_trading_db",
                "user": "postgres",
                "password": "secure_password",
                "host": "localhost",
                "port": 5432
            })
            self.trading_db = TradingDB(db_config)
            await self.trading_db.connect()

            # Inicializar procesadores
            self.analyzer_processor = AnalyzerProcessor(config, self.redis_client)
            self.capital_processor = CapitalProcessor(config)
            self.exchange_processor = ExchangeProcessor(config, self.nucleus)
            await self.exchange_processor.inicializar()
            self.execution_processor = ExecutionProcessor(config, self.redis_client)
            self.macro_processor = MacroProcessor(config, self.redis_client)
            self.monitor_processor = MonitorProcessor(config, self.redis_client)
            self.predictor_processor = PredictorProcessor(config, self.redis_client)
            await self.predictor_processor.inicializar()
            self.strategy = MomentumStrategy(self.capital)

            # Obtener pares con mayor volumen por exchange
            activos = await self.exchange_processor.detectar_disponibles()
            for ex in activos:
                exchange_name = ex["name"]
                client = ex["client"]
                top_pairs = await self.exchange_processor.obtener_top_activos(client)
                self.trading_pairs_by_exchange[exchange_name] = top_pairs
                self.logger.info(f"Pares configurados para {exchange_name}: {top_pairs}")

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
                return await self._execute_trade(params.get("exchange"), params.get("pair"), params.get("side"))
            else:
                return {"status": "error", "message": f"Acción no soportada: {action}"}
        except Exception as e:
            self.logger.error(f"[CryptoTrading] Error al manejar comando: {e}")
            return {"status": "error", "message": str(e)}

    async def _monitor_loop(self):
        """Bucle de monitoreo que verifica el mercado cada 3 minutos."""
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
                    # Obtener datos reales
                    macro_data = await self.alpha_vantage.fetch_macro_data()
                    for block in self.monitor_blocks:
                        result = await block.procesar(0.5)
                        if result["status"] != "success":
                            self.logger.warning(f"Error en monitoreo: {result['motivo']}")
                            continue

                        # Priorizar pares con mayor volatilidad
                        volatilidad = result["result"]["volatilidad"]
                        prioritized_pairs = []
                        for exchange, pairs in self.trading_pairs_by_exchange.items():
                            for pair in pairs:
                                for v in volatilidad:
                                    if v["symbol"] == pair and v["alerta"]:
                                        prioritized_pairs.append((exchange, pair, v["volatilidad"]))
                                        break
                            # Añadir el resto de los pares no volátiles
                            for pair in pairs:
                                if not any(p[1] == pair for p in prioritized_pairs):
                                    prioritized_pairs.append((exchange, pair, 0.01))

                        # Obtener datos de criptomonedas para cada par, por exchange
                        for exchange, pair, vol in prioritized_pairs:
                            crypto_data = await self.coinmarketcap.fetch_crypto_data(pair)
                            combined_crypto_data = crypto_data

                            # Obtener precios históricos (simulados por ahora)
                            prices = [50000 + i * 100 for i in range(50)]  # En producción, obtener de exchange
                            sentiment = self.strategy.calculate_momentum(macro_data, combined_crypto_data, prices, vol)
                            side = self.strategy.decide_trade(exchange, pair, sentiment, vol)

                            if side == "pending":
                                continue

                            trade_result = await self._execute_trade(exchange, pair, side)
                            if trade_result["status"] == "success":
                                self.open_trades[f"{exchange}:{pair}"] = trade_result["result"]

                await asyncio.sleep(180)  # Esperar 3 minutos
            except Exception as e:
                self.logger.error(f"[CryptoTrading] Error en bucle de monitoreo: {e}")
                await asyncio.sleep(180)  # Continuar tras error

    async def _monitor_open_trades(self):
        """Monitorea operaciones abiertas fuera del horario de trading."""
        for trade_id, trade in list(self.open_trades.items()):
            try:
                exchange, pair = trade_id.split(":")
                self.logger.info(f"[CryptoTrading] Monitoreando operación abierta para {exchange}:{pair}: {trade}")
                # Simulación de precio actual (en producción, obtener del exchange)
                current_price = 50000
                if (trade["tipo"] == "buy" and current_price >= trade["take_profit"]) or \
                   (trade["tipo"] == "buy" and current_price <= trade["stop_loss"]) or \
                   (trade["tipo"] == "sell" and current_price <= trade["take_profit"]) or \
                   (trade["tipo"] == "sell" and current_price >= trade["stop_loss"]):
                    await self._close_trade(exchange, pair, trade)
            except Exception as e:
                self.logger.error(f"[CryptoTrading] Error al monitorear operación abierta para {trade_id}: {e}")

    async def _close_trade(self, exchange: str, pair: str, trade: dict):
        """Cierra una operación abierta y actualiza el capital."""
        trade_id = f"{exchange}:{pair}"
        self.logger.info(f"[CryptoTrading] Cerrando operación para {trade_id}")
        trade["status"] = "closed"
        trade["close_timestamp"] = datetime.datetime.utcnow().isoformat()
        await self.trading_db.save_order(
            exchange,
            trade["orden_id"],
            pair,
            "spot",
            "closed",
            datetime.datetime.utcnow().timestamp()
        )
        # Actualizar capital (simulado por ahora)
        profit = trade["cantidad"] * (50000 - trade["precio"])  # Simulado
        self.capital += trade["cantidad"] + profit
        self.strategy.update_capital(self.capital)
        self.execution_processor.update_capital(self.capital)
        del self.open_trades[trade_id]
        await self.nucleus.publicar_alerta({
            "tipo": "operacion_cerrada",
            "plugin_id": "crypto_trading",
            "exchange": exchange,
            "pair": pair,
            "timestamp": trade["close_timestamp"]
        })

    async def _execute_trade(self, exchange: str, pair: str, side: str) -> Dict[str, Any]:
        """Ejecuta una operación de trading usando el TradingBlock y actualiza el capital."""
        for block in self.trading_blocks:
            result = await block.procesar(0.5, exchange, pair, side, paper_mode=self.paper_mode)
            if result["status"] == "success":
                if side == "buy":
                    self.capital -= result["result"]["cantidad"]
                elif side == "sell":
                    self.capital += result["result"]["cantidad"]
                self.strategy.update_capital(self.capital)
                self.execution_processor.update_capital(self.capital)
                return result
        return {"status": "error", "motivo": "No se pudo ejecutar la operación"}

    async def detener(self):
        """Detiene el plugin CryptoTrading."""
        await self.exchange_processor.close()
        await self.trading_db.disconnect()
        self.logger.info("[CryptoTrading] Plugin detenido")
