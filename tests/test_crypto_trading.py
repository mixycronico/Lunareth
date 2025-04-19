#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/tests/test_crypto_trading.py
Pruebas simuladas para el plugin CryptoTrading.
"""
import unittest
import asyncio
import json
import torch
import psycopg2
import jwt
from unittest.mock import AsyncMock, patch
from plugins.crypto_trading.processors.exchange_processor import ExchangeProcessor
from plugins.crypto_trading.processors.capital_processor import CapitalProcessor
from plugins.crypto_trading.processors.settlement_processor import SettlementProcessor
from plugins.crypto_trading.processors.macro_processor import MacroProcessor
from plugins.crypto_trading.processors.monitor_processor import MonitorProcessor
from plugins.crypto_trading.processors.predictor_processor import PredictorProcessor
from plugins.crypto_trading.processors.analyzer_processor import AnalyzerProcessor
from plugins.crypto_trading.processors.execution_processor import ExecutionProcessor
from plugins.crypto_trading.processors.user_processor import UserProcessor
from corec.core import aioredis

class TestCryptoTrading(unittest.TestCase):
    def setUp(self):
        self.config = {
            "crypto_trading": {
                "exchange_config": {
                    "exchanges": [
                        {"name": "binance", "api_key": "test_key", "api_secret": "test_secret", "symbols": ["BTC/USDT"]},
                        {"name": "kraken", "api_key": "test_key", "api_secret": "test_secret", "symbols": ["BTC/USD"]}
                    ],
                    "fetch_interval": 300,
                    "circuit_breaker": {"max_failures": 2, "reset_timeout": 60}
                },
                "capital_config": {
                    "min_contribution": 100,
                    "max_active_ratio": 0.6,
                    "phases": [
                        {"name": "conservative", "min": 0, "max": 10000, "risk_per_trade": 0.01},
                        {"name": "moderate", "min": 10000, "max": 50000, "risk_per_trade": 0.02}
                    ],
                    "circuit_breaker": {"max_failures": 2, "reset_timeout": 60}
                },
                "settlement_config": {
                    "settlement_time": "23:59",
                    "circuit_breaker": {"max_failures": 2, "reset_timeout": 60}
                },
                "macro_config": {
                    "symbols": ["^GSPC", "^VIX"],
                    "altcoin_symbols": ["SOL", "ADA"],
                    "update_interval": 300,
                    "api_keys": {
                        "alpha_vantage": "test_key",
                        "coinmarketcap": "test_key",
                        "newsapi": "test_key"
                    },
                    "circuit_breaker": {"max_failures": 2, "reset_timeout": 60}
                },
                "monitor_config": {
                    "symbols": ["BTC/USDT", "ETH/USDT"],
                    "update_interval": 60,
                    "circuit_breaker": {"max_failures": 2, "reset_timeout": 60}
                },
                "predictor_config": {
                    "lstm_window": 60,
                    "lstm_hidden_size": 50,
                    "lstm_layers": 2,
                    "max_datos": 1000,
                    "model_path": "test_model.pth",
                    "retrain_interval": 86400,
                    "circuit_breaker": {"max_failures": 2, "reset_timeout": 60}
                },
                "analyzer_config": {
                    "analysis_interval": 300,
                    "auto_execute": true,
                    "circuit_breaker": {"max_failures": 2, "reset_timeout": 60}
                },
                "execution_config": {
                    "risk_per_trade": 0.02,
                    "take_profit": 0.05,
                    "stop_loss": 0.02,
                    "circuit_breaker": {"max_failures": 2, "reset_timeout": 60}
                },
                "user_config": {
                    "jwt_secret": "test_secret",
                    "circuit_breaker": {"max_failures": 2, "reset_timeout": 60}
                },
                "db_config": {
                    "dbname": "test_db",
                    "user": "test_user",
                    "password": "test_password",
                    "host": "localhost",
                    "port": "5432"
                }
            }
        }
        self.redis_client = AsyncMock()
        self.exchange_processor = ExchangeProcessor(self.config, self.redis_client)
        self.capital_processor = CapitalProcessor(self.config, self.redis_client)
        self.settlement_processor = SettlementProcessor(self.config, self.redis_client)
        self.macro_processor = MacroProcessor(self.config, self.redis_client)
        self.monitor_processor = MonitorProcessor(self.config, self.redis_client)
        self.predictor_processor = PredictorProcessor(self.config, self.redis_client)
        self.analyzer_processor = AnalyzerProcessor(self.config, self.redis_client)
        self.execution_processor = ExecutionProcessor(self.config, self.redis_client)
        self.user_processor = UserProcessor(self.config, self.redis_client)
        self.loop = asyncio.get_event_loop()

    async def test_exchange_inicializar(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.connect", AsyncMock(return_value=True)):
            await self.exchange_processor.inicializar()
            self.assertIsNotNone(self.exchange_processor.plugin_db)
            self.assertEqual(len(self.exchange_processor.circuit_breakers), 2)

    async def test_exchange_fetch_spot_price_binance(self):
        with patch("aiohttp.ClientSession.get", AsyncMock(return_value=AsyncMock(status=200, json=AsyncMock(return_value={"price": "50000"})))):
            async with aiohttp.ClientSession() as session:
                result = await self.exchange_processor.fetch_spot_price(
                    {"name": "binance", "symbols": ["BTC/USDT"]}, "BTC/USDT", session
                )
                self.assertEqual(result["exchange"], "binance")
                self.assertEqual(result["price"], 50_000.0)

    async def test_exchange_circuit_breaker(self):
        with patch("aiohttp.ClientSession.get", AsyncMock(side_effect=Exception("API error"))):
            async with aiohttp.ClientSession() as session:
                await self.exchange_processor.fetch_spot_price(
                    {"name": "binance", "symbols": ["BTC/USDT"]}, "BTC/USDT", session
                )
                await self.exchange_processor.fetch_spot_price(
                    {"name": "binance", "symbols": ["BTC/USDT"]}, "BTC/USDT", session
                )
                self.assertTrue(self.exchange_processor.circuit_breakers["binance"].tripped)

    async def test_capital_inicializar(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.connect", AsyncMock(return_value=True)):
            with patch("plugins.crypto_trading.utils.db.TradingDB.get_pool_total", AsyncMock(return_value=1000.0)):
                with patch("plugins.crypto_trading.utils.db.TradingDB.get_users", AsyncMock(return_value={"user1": 500.0})):
                    with patch("plugins.crypto_trading.utils.db.TradingDB.get_active_capital", AsyncMock(return_value=200.0)):
                        await self.capital_processor.inicializar()
                        self.assertEqual(self.capital_processor.pool, 1000.0)
                        self.assertEqual(self.capital_processor.users, {"user1": 500.0})
                        self.assertEqual(self.capital_processor.active_capital, 200.0)

    async def test_capital_add_contribution(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.save_contribution", AsyncMock()):
            with patch("plugins.crypto_trading.utils.db.TradingDB.update_pool", AsyncMock()):
                result = await self.capital_processor.add_contribution("user1", 200.0)
                self.assertTrue(result)
                self.assertEqual(self.capital_processor.users["user1"], 200.0)
                self.assertEqual(self.capital_processor.pool, 200.0)

    async def test_capital_process_withdrawal(self):
        self.capital_processor.users = {"user1": 500.0}
        self.capital_processor.pool = 500.0
        with patch("plugins.crypto_trading.utils.db.TradingDB.save_withdrawal", AsyncMock()):
            with patch("plugins.crypto_trading.utils.db.TradingDB.update_pool", AsyncMock()):
                result = await self.capital_processor.process_withdrawal("user1", 300.0)
                self.assertTrue(result)
                self.assertEqual(self.capital_processor.users["user1"], 200.0)
                self.assertEqual(self.capital_processor.pool, 200.0)

    async def test_capital_assign_capital(self):
        self.capital_processor.pool = 10_000.0
        self.capital_processor.active_capital = 2000.0
        with patch("plugins.crypto_trading.utils.db.TradingDB.update_active_capital", AsyncMock()):
            await self.capital_processor.assign_capital()
            self.redis_client.xadd.assert_called()
            self.assertGreater(self.capital_processor.active_capital, 2000.0)

    async def test_settlement_inicializar(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.connect", AsyncMock(return_value=True)):
            await self.settlement_processor.inicializar()
            self.assertIsNotNone(self.settlement_processor.plugin_db)

    async def test_settlement_consolidate_results(self):
        self.settlement_processor.trading_results = [
            {"profit": 100.0, "symbol": "BTC/USDT", "exchange": "binance"},
            {"profit": -50.0, "symbol": "ETH/USDT", "exchange": "kraken"}
        ]
        with patch("plugins.crypto_trading.utils.db.TradingDB.get_pool_total", AsyncMock(return_value=1000.0)):
            with patch("plugins.crypto_trading.utils.db.TradingDB.get_active_capital", AsyncMock(return_value=200.0)):
                with patch("plugins.crypto_trading.utils.db.TradingDB.get_users", AsyncMock(return_value={"user1": 500.0, "user2": 500.0})):
                    with patch("plugins.crypto_trading.utils.db.TradingDB.save_report", AsyncMock()):
                        result = await self.settlement_processor.consolidate_results()
                        self.assertEqual(result["status"], "ok")
                        self.assertEqual(result["report"]["total_profit"], 50.0)
                        self.assertEqual(result["report"]["total_trades"], 2)
                        self.assertEqual(result["report"]["roi_percent"], 5.0)
                        self.redis_client.xadd.assert_called()

    async def test_settlement_circuit_breaker(self):
        self.settlement_processor.circuit_breaker.failure_count = 2
        self.settlement_processor.circuit_breaker.register_failure()
        self.assertTrue(self.settlement_processor.circuit_breaker.tripped)
        result = await self.settlement_processor.consolidate_results()
        self.assertEqual(result["status"], "error")

    async def test_macro_inicializar(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.connect", AsyncMock(return_value=True)):
            await self.macro_processor.inicializar()
            self.assertIsNotNone(self.macro_processor.plugin_db)
            self.assertEqual(len(self.macro_processor.circuit_breakers), 6)

    async def test_macro_fetch_alpha_vantage(self):
        with patch("aiohttp.ClientSession.get", AsyncMock(return_value=AsyncMock(status=200, json=AsyncMock(return_value={
            "Global Quote": {"05. price": "5000", "10. change percent": "1.5%"}
        })))):
            result = await self.macro_processor.fetch_alpha_vantage("^GSPC")
            self.assertEqual(result["price"], 5000.0)
            self.assertEqual(result["change_percent"], 1.5)

    async def test_macro_fetch_dxy(self):
        with patch("aiohttp.ClientSession.get", AsyncMock(return_value=AsyncMock(status=200, json=AsyncMock(return_value={
            "Realtime Currency Exchange Rate": {"5. Exchange Rate": "100"}
        })))):
            result = await self.macro_processor.fetch_dxy()
            self.assertEqual(result["price"], 100.0)
            self.assertEqual(result["change_percent"], 0.0)

    async def test_macro_circuit_breaker(self):
        self.macro_processor.circuit_breakers["^GSPC"].failure_count = 2
        self.macro_processor.circuit_breakers["^GSPC"].register_failure()
        self.assertTrue(self.macro_processor.circuit_breakers["^GSPC"].tripped)
        result = await self.macro_processor.fetch_alpha_vantage("^GSPC")
        self.assertEqual(result, {})

    async def test_monitor_inicializar(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.connect", AsyncMock(return_value=True)):
            await self.monitor_processor.inicializar()
            self.assertIsNotNone(self.monitor_processor.plugin_db)

    async def test_monitor_prices(self):
        self.monitor_processor.price_cache = {
            "BTC/USDT": {
                "data": {
                    "binance": {"price": 50_000.0, "volume": 1000, "timestamp": datetime.utcnow().timestamp()},
                    "kraken": {"price": 51_000.0, "volume": 500, "timestamp": datetime.utcnow().timestamp()}
                },
                "expires": datetime.utcnow().timestamp() + 300
            }
        }
        with patch("plugins.crypto_trading.utils.db.TradingDB.save_price", AsyncMock()):
            await self.monitor_processor.monitor_prices()
            self.redis_client.xadd.assert_called()
            call_args = self.redis_client.xadd.call_args
            self.assertEqual(call_args[0][0], "market_data")
            data = json.loads(zstd.decompress(call_args[0][1]["data"]))
            self.assertEqual(data["symbol"], "BTC/USDT")
            self.assertAlmostEqual(data["price"], 50_333.33, places=2)

    async def test_monitor_altcoins_update(self):
        mensaje = {"tipo": "macro_data", "altcoins": ["SOL", "ADA"]}
        await self.monitor_processor.manejar_evento(mensaje)
        self.assertEqual(self.monitor_processor.altcoins, ["SOL", "ADA"])

    async def test_monitor_circuit_breaker(self):
        self.monitor_processor.circuit_breaker.failure_count = 2
        self.monitor_processor.circuit_breaker.register_failure()
        self.assertTrue(self.monitor_processor.circuit_breaker.tripped)
        await self.monitor_processor.monitor_prices()
        self.redis_client.xadd.assert_not_called()

    async def test_predictor_inicializar(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.connect", AsyncMock(return_value=True)):
            with patch("torch.load", AsyncMock(side_effect=FileNotFoundError)):
                await self.predictor_processor.inicializar()
                self.assertIsNotNone(self.predictor_processor.plugin_db)
                self.assertIsNotNone(self.predictor_processor.model)

    async def test_predictor_procesar(self):
        self.predictor_processor.price_history = {
            "BTC/USDT": [50_000.0] * 60
        }
        with patch("plugins.crypto_trading.utils.db.TradingDB.save_prediction", AsyncMock()):
            result = await self.predictor_processor.procesar(
                {"symbol": "BTC/USDT", "actual_value": 51_000.0},
                {"tipo": "market_data", "instance_id": "test", "nano_id": "predictor_temporal", "timestamp": 1234567890}
            )
            self.assertEqual(result["estado"], "ok")
            self.assertIn("prediction", result)
            self.redis_client.xadd.assert_called()

    async def test_predictor_circuit_breaker(self):
        self.predictor_processor.circuit_breaker.failure_count = 2
        self.predictor_processor.circuit_breaker.register_failure()
        self.assertTrue(self.predictor_processor.circuit_breaker.tripped)
        result = await self.predictor_processor.procesar(
            {"symbol": "BTC/USDT", "valores": [50_000.0] * 60},
            {"tipo": "market_data", "instance_id": "test", "nano_id": "predictor_temporal", "timestamp": 1234567890}
        )
        self.assertEqual(result["estado"], "error")

    async def test_analyzer_inicializar(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.connect", AsyncMock(return_value=True)):
            await self.analyzer_processor.inicializar()
            self.assertIsNotNone(self.analyzer_processor.plugin_db)

    async def test_analyzer_analyze_system(self):
        self.analyzer_processor.metrics_cache = {
            "predictor_temporal": {"mse": 20},
            "settlement_data": {"roi_percent": 5, "total_trades": 10, "profits": [100, -50, 200]},
            "capital_data": {"pool_total": 1000},
            "alert_data": [{"severity": "high"}, {"severity": "high"}, {"severity": "high"}],
            "eventos": {"nodes": 4},
            "auditoria": {"load": 0.7},
            "macro_data": {"vix_price": 25, "dxy_change_percent": 0.6}
        }
        with patch("plugins.crypto_trading.utils.db.TradingDB.save_insight", AsyncMock()):
            await self.analyzer_processor.analyze_system()
            self.redis_client.xadd.assert_called()
            call_args = self.redis_client.xadd.call_args
            self.assertEqual(call_args[0][0], "crypto_trading_data")
            data = json.loads(zstd.decompress(call_args[0][1]["data"]))
            self.assertIn("recommendations", data)
            self.assertGreater(len(data["recommendations"]), 0)

    async def test_analyzer_sharpe_ratio(self):
        profits = [100, -50, 200, 150]
        sharpe = await self.analyzer_processor.calculate_sharpe_ratio(profits)
        self.assertGreater(sharpe, 0.0)

    async def test_analyzer_circuit_breaker(self):
        self.analyzer_processor.circuit_breaker.failure_count = 2
        self.analyzer_processor.circuit_breaker.register_failure()
        self.assertTrue(self.analyzer_processor.circuit_breaker.tripped)
        await self.analyzer_processor.analyze_system()
        self.redis_client.xadd.assert_not_called()

    async def test_execution_inicializar(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.connect", AsyncMock(return_value=True)):
            await self.execution_processor.inicializar()
            self.assertIsNotNone(self.execution_processor.plugin_db)
            self.assertEqual(len(self.execution_processor.circuit_breakers), 2)

    async def test_execution_place_order(self):
        with patch("aiohttp.ClientSession.request", AsyncMock(return_value=AsyncMock(status=200, json=AsyncMock(return_value={"orderId": "12345"})))):
            with patch("plugins.crypto_trading.utils.db.TradingDB.save_order", AsyncMock()):
                result = await self.execution_processor.place_order(
                    {"name": "binance", "api_key": "test_key", "api_secret": "test_secret"},
                    "BTC/USDT",
                    "buy",
                    0.01,
                    "spot",
                    50_000.0
                )
                self.assertEqual(result["exchange"], "binance")
                self.assertEqual(result["order_id"], "12345")
                self.redis_client.xadd.assert_called()

    async def test_execution_backtest(self):
        with patch("psycopg2.connect", AsyncMock()):
            with patch("psycopg2.connect.cursor", AsyncMock(return_value=AsyncMock(fetchall=AsyncMock(return_value=[(50_000.0, 1234567890)] * 1000)))):
                result = await self.execution_processor.run_backtest({"symbol": "BTC/USDT", "trades": 20})
                self.assertEqual(result["estado"], "ok")
                self.assertIn("roi", result)
                self.assertIn("sharpe_ratio", result)

    async def test_execution_circuit_breaker(self):
        self.execution_processor.circuit_breakers["binance"].failure_count = 2
        self.execution_processor.circuit_breakers["binance"].register_failure()
        self.assertTrue(self.execution_processor.circuit_breakers["binance"].tripped)
        result = await self.execution_processor.place_order(
            {"name": "binance", "api_key": "test_key", "api_secret": "test_secret"},
            "BTC/USDT",
            "buy",
            0.01,
            "spot",
            50_000.0
        )
        self.assertEqual(result, {})

    async def test_user_inicializar(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.connect", AsyncMock(return_value=True)):
            await self.user_processor.inicializar()
            self.assertIsNotNone(self.user_processor.plugin_db)

    async def test_user_register(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.save_user", AsyncMock()):
            result = await self.user_processor.procesar_usuario({
                "action": "register",
                "user_id": "user1",
                "email": "user1@example.com",
                "password": "password123",
                "name": "User One",
                "role": "user",
                "notification_preferences": {"email": True},
                "requester_id": "admin1"
            })
            self.assertEqual(result["estado"], "ok")
            self.assertEqual(result["user_id"], "user1")
            self.assertIn("jwt_token", result)
            self.redis_client.xadd.assert_called()

    async def test_user_login(self):
        hashed_password = await self.user_processor.hash_password("password123")
        with patch("plugins.crypto_trading.utils.db.TradingDB.get_user", AsyncMock(return_value={
            "user_id": "user1",
            "password": hashed_password,
            "role": "user"
        })):
            result = await self.user_processor.procesar_usuario({
                "action": "login",
                "user_id": "user1",
                "password": "password123"
            })
            self.assertEqual(result["estado"], "ok")
            self.assertEqual(result["user_id"], "user1")
            self.assertIn("jwt_token", result)

    async def test_user_check_permission(self):
        with patch("plugins.crypto_trading.utils.db.TradingDB.get_user", AsyncMock(return_value={"user_id": "user1", "role": "user"})):
            result = await self.user_processor.check_permission("user1", "view_reports")
            self.assertTrue(result)
            result = await self.user_processor.check_permission("user1", "manage_users")
            self.assertFalse(result)

    async def test_user_circuit_breaker(self):
        self.user_processor.circuit_breaker.failure_count = 2
        self.user_processor.circuit_breaker.register_failure()
        self.assertTrue(self.user_processor.circuit_breaker.tripped)
        result = await self.user_processor.procesar_usuario({
            "action": "register",
            "user_id": "user1",
            "email": "user1@example.com",
            "password": "password123",
            "name": "User One",
            "role": "user",
            "notification_preferences": {"email": True},
            "requester_id": "admin1"
        })
        self.assertEqual(result["estado"], "error")

    def test_all(self):
        async def run_tests():
            await self.test_exchange_inicializar()
            await self.test_exchange_fetch_spot_price_binance()
            await self.test_exchange_circuit_breaker()
            await self.test_capital_inicializar()
            await self.test_capital_add_contribution()
            await self.test_capital_process_withdrawal()
            await self.test_capital_assign_capital()
            await self.test_settlement_inicializar()
            await self.test_settlement_consolidate_results()
            await self.test_settlement_circuit_breaker()
            await self.test_macro_inicializar()
            await self.test_macro_fetch_alpha_vantage()
            await self.test_macro_fetch_dxy()
            await self.test_macro_circuit_breaker()
            await self.test_monitor_inicializar()
            await self.test_monitor_prices()
            await self.test_monitor_altcoins_update()
            await self.test_monitor_circuit_breaker()
            await self.test_predictor_inicializar()
            await self.test_predictor_procesar()
            await self.test_predictor_circuit_breaker()
            await self.test_analyzer_inicializar()
            await self.test_analyzer_analyze_system()
            await self.test_analyzer_sharpe_ratio()
            await self.test_analyzer_circuit_breaker()
            await self.test_execution_inicializar()
            await self.test_execution_place_order()
            await self.test_execution_backtest()
            await self.test_execution_circuit_breaker()
            await self.test_user_inicializar()
            await self.test_user_register()
            await self.test_user_login()
            await self.test_user_check_permission()
            await self.test_user_circuit_breaker()
        self.loop.run_until_complete(run_tests())

if __name__ == "__main__":
    unittest.main()