
# tests/test_crypto_trading_full.py

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from plugins.crypto_trading.main import CryptoTrading
from plugins.crypto_trading.config import CryptoTradingConfig
from plugins.crypto_trading.utils.db import TradingDB
from plugins.crypto_trading.processors.analyzer_processor import AnalyzerProcessor
from plugins.crypto_trading.processors.execution_processor import ExecutionProcessor
from plugins.crypto_trading.processors.predictor_processor import PredictorProcessor
from plugins.crypto_trading.processors.settlement_processor import SettlementProcessor
from plugins.crypto_trading.processors.monitor_processor import MonitorProcessor
from plugins.crypto_trading.processors.macro_processor import MacroProcessor

@pytest.fixture
def fake_config():
    return CryptoTradingConfig(**{
        "exchange_config": {
            "exchanges": [{"name": "binance", "api_key": "key", "api_secret": "secret", "symbols": ["BTC/USDT"]}],
            "fetch_interval": 60,
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "capital_config": {
            "min_contribution": 100,
            "max_active_ratio": 0.6,
            "phases": [
                {"name": "aggressive", "min": 0, "max": 1000000, "risk_per_trade": 0.02}
            ],
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "settlement_config": {
            "settlement_time": "23:59",
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "macro_config": {
            "symbols": ["^GSPC", "^IXIC"],
            "altcoin_symbols": ["SOL"],
            "update_interval": 300,
            "api_keys": {"alpha_vantage": "key", "coinmarketcap": "key", "newsapi": "key"},
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "monitor_config": {
            "symbols": ["BTC/USDT"],
            "update_interval": 60,
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "predictor_config": {
            "lstm_window": 60,
            "lstm_hidden_size": 50,
            "lstm_layers": 2,
            "max_datos": 1000,
            "model_path": "model.pth",
            "retrain_interval": 86400,
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "analyzer_config": {
            "analysis_interval": 300,
            "auto_execute": True,
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "execution_config": {
            "risk_per_trade": 0.02,
            "take_profit": 0.05,
            "stop_loss": 0.02,
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "user_config": {
            "jwt_secret": "secret",
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "db_config": {
            "dbname": "test_db",
            "user": "user",
            "password": "pass",
            "host": "localhost",
            "port": 5432
        }
    })

@pytest.mark.asyncio
async def test_crypto_trading_full_cycle(fake_config):
    nucleus = MagicMock()
    nucleus.redis_client = AsyncMock()
    db = TradingDB(db_config=fake_config.db_config)
    db.connect = AsyncMock(return_value=True)
    db.save_price = AsyncMock()
    db.save_order = AsyncMock()
    db.get_pool_total = AsyncMock(return_value=10000)

    crypto_trading = CryptoTrading(nucleus, fake_config)
    crypto_trading.db = db

    await crypto_trading.inicializar()

    await crypto_trading.analyzer.procesar()
    await crypto_trading.predictor.procesar()
    await crypto_trading.execution.procesar()
    await crypto_trading.settlement.procesar()
    await crypto_trading.monitor.procesar()
    await crypto_trading.macro.procesar()

    assert crypto_trading.analyzer is not None
    assert crypto_trading.predictor is not None
    assert crypto_trading.execution is not None
    assert crypto_trading.settlement is not None
    assert crypto_trading.monitor is not None
    assert crypto_trading.macro is not None
