import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from plugins.crypto_trading.main import CryptoTrading
from plugins.crypto_trading.config_loader import load_config_dict, AppConfig
from plugins.crypto_trading.processors.orchestrator_processor import OrchestratorProcessor
from plugins.crypto_trading.processors.exchange_processor import ExchangeProcessor
from plugins.crypto_trading.processors.execution_processor import ExecutionProcessor
from plugins.crypto_trading.processors.macro_processor import MacroProcessor
from plugins.crypto_trading.processors.ia_analysis_processor import IAAnalisisProcessor
from plugins.crypto_trading.processors.predictor_processor import PredictorProcessor
from plugins.crypto_trading.processors.settlement_processor import SettlementProcessor
from plugins.crypto_trading.processors.capital_processor import CapitalProcessor
from plugins.crypto_trading.processors.monitor_processor import MonitorProcessor
from plugins.crypto_trading.strategies.momentum_strategy import MomentumStrategy
from plugins.crypto_trading.utils.db import TradingDB
from plugins.crypto_trading.utils.helpers import CircuitBreaker
from plugins.crypto_trading.blocks.trading_block import TradingBlock
from plugins.crypto_trading.blocks.monitor_block import MonitorBlock
from plugins.crypto_trading.data.alpha_vantage_fetcher import AlphaVantageFetcher
from plugins.crypto_trading.data.coinmarketcap_fetcher import CoinMarketCapFetcher
import numpy as np
import torch

# --- Fixtures Adicionales para el Plugin CryptoTrading ---

@pytest.fixture
def mock_config_trading():
    """Fixture para la configuración del plugin de trading."""
    return {
        "exchange_config": {
            "exchanges": [
                {"name": "binance", "api_key": "key", "api_secret": "secret", "symbols": ["BTC/USDT"]},
                {"name": "kucoin", "api_key": "key", "api_secret": "secret", "symbols": ["BTC/USDT"]}
            ],
            "fetch_interval": 300,
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "capital_config": {
            "min_contribution": 100,
            "max_active_ratio": 0.6,
            "phases": [
                {"name": "conservative", "min": 0, "max": 10000, "risk_per_trade": 0.01},
                {"name": "moderate", "min": 10000, "max": 50000, "risk_per_trade": 0.02},
                {"name": "aggressive", "min": 50000, "max": 1000000, "risk_per_trade": 0.03}
            ],
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900},
            "total_capital": 1000.0
        },
        "settlement_config": {
            "settlement_time": "23:59",
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "macro_config": {
            "symbols": ["^GSPC", "^IXIC"],
            "altcoin_symbols": ["SOL", "ADA"],
            "update_interval": 300,
            "api_keys": {
                "alpha_vantage": "mock_key",
                "coinmarketcap": "mock_key",
                "newsapi": "mock_key"
            },
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
            "model_path": "mock_model.pth",
            "retrain_interval": 86400,
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "analyzer_config": {
            "analysis_interval": 300,
            "auto_execute": True,
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900},
            "volatility_threshold": 0.025
        },
        "execution_config": {
            "risk_per_trade": 0.02,
            "take_profit": 0.05,
            "stop_loss": 0.02,
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "user_config": {
            "jwt_secret": "secure_secret",
            "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
        },
        "db_config": {
            "dbname": "trading_db",
            "user": "trading_user",
            "password": "secure_password",
            "host": "localhost",
            "port": "5432"
        },
        "cb_max_failures": 3,
        "cb_reset_timeout": 900,
        "total_capital": 1000.0,
        "paper_mode": True,
        "symbols": ["BTC/USDT"],
        "volatility_threshold": 0.025
    }

@pytest.fixture
def mock_ccxt_exchange():
    """Fixture para simular un cliente de ccxt."""
    exchange = MagicMock()
    exchange.fetch_status = AsyncMock(return_value={"status": "ok"})
    exchange.fetch_tickers = AsyncMock(return_value={
        "BTC/USDT": {"quoteVolume": 1000000},
        "ETH/USDT": {"quoteVolume": 500000}
    })
    exchange.close = AsyncMock(return_value=None)
    return exchange

@pytest.fixture
def mock_market_data():
    """Fixture para simular datos de mercado."""
    return {
        "macro": {
            "sp500": 0.02,
            "nasdaq": 0.01,
            "dxy": 0.0,
            "gold": 0.01,
            "oil": 0.01,
            "vix": 0.01
        },
        "crypto": {
            "BTC/USDT": {"volume": 1000000, "market_cap": 1000000000, "price_change": 0.01},
            "ETH/USDT": {"volume": 500000, "market_cap": 500000000, "price_change": 0.01}
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@pytest.fixture
def mock_crypto_trading(mock_config_trading, nucleus, mock_redis):
    """Fixture para el plugin CryptoTrading."""
    plugin = CryptoTrading()
    plugin.nucleus = nucleus
    plugin.redis_client = mock_redis
    plugin.config = mock_config_trading
    return plugin

# --- Pruebas de Configuración ---

@pytest.mark.asyncio
async def test_config_load_valid(mock_config_trading):
    """Prueba la carga y validación de la configuración."""
    config = load_config_dict()
    assert config["exchange_config"]["exchanges"][0]["name"] == "binance"
    assert config["capital_config"]["total_capital"] == 1000.0
    assert config["settlement_config"]["settlement_time"] == "23:59"

@pytest.mark.asyncio
async def test_config_load_invalid():
    """Prueba la carga de una configuración inválida."""
    invalid_config = {
        "crypto_trading": {
            "exchange_config": {
                "exchanges": [],
                "fetch_interval": -300,  # Valor inválido
                "circuit_breaker": {"max_failures": 3, "reset_timeout": 900}
            }
        }
    }
    with pytest.raises(ValidationError):
        AppConfig(**invalid_config)

# --- Pruebas de OrchestratorProcessor ---

@pytest.mark.asyncio
async def test_orchestrator_processor_initialize(mock_config_trading, nucleus, mock_redis, mock_db_pool):
    """Prueba la inicialización del OrchestratorProcessor."""
    orchestrator = OrchestratorProcessor(mock_config_trading, nucleus, mock_redis)
    with patch.object(orchestrator.trading_db, "connect", new=AsyncMock()), \
         patch.object(orchestrator, "load_processor", new=AsyncMock()), \
         patch.object(orchestrator.processors["exchange"], "inicializar", new=AsyncMock()), \
         patch.object(orchestrator.processors["exchange"], "detectar_disponibles", return_value=[]), \
         patch.object(orchestrator.processors["settlement"], "restore_state", new=AsyncMock()), \
         patch.object(orchestrator.logger, "info") as mock_logger:
        await orchestrator.initialize()
        assert mock_logger.called
        assert len(orchestrator.trading_blocks) == 1
        assert len(orchestrator.monitor_blocks) == 1

@pytest.mark.asyncio
async def test_orchestrator_processor_initialize_error(mock_config_trading, nucleus, mock_redis):
    """Prueba la inicialización con error del OrchestratorProcessor."""
    orchestrator = OrchestratorProcessor(mock_config_trading, nucleus, mock_redis)
    with patch.object(orchestrator.trading_db, "connect", side_effect=Exception("DB Error")), \
         patch.object(orchestrator.logger, "error") as mock_logger, \
         patch.object(nucleus, "publicar_alerta", new=AsyncMock()) as mock_alerta:
        with pytest.raises(Exception):
            await orchestrator.initialize()
        assert mock_logger.called
        assert mock_alerta.called

# --- Pruebas de ExchangeProcessor ---

@pytest.mark.asyncio
async def test_exchange_processor_initialize(mock_config_trading, nucleus, mock_ccxt_exchange):
    """Prueba la inicialización del ExchangeProcessor."""
    strategy = MagicMock()
    execution_processor = MagicMock()
    settlement_processor = MagicMock()
    monitor_blocks = []
    processor = ExchangeProcessor(mock_config_trading["exchange_config"], nucleus, strategy, execution_processor, settlement_processor, monitor_blocks)
    with patch("ccxt.async_support.binance", return_value=mock_ccxt_exchange), \
         patch("ccxt.async_support.kucoin", return_value=mock_ccxt_exchange):
        await processor.inicializar()
        assert "binance" in processor.exchange_clients
        assert "kucoin" in processor.exchange_clients

@pytest.mark.asyncio
async def test_exchange_processor_detectar_disponibles(mock_config_trading, nucleus, mock_ccxt_exchange):
    """Prueba la detección de exchanges disponibles."""
    strategy = MagicMock()
    execution_processor = MagicMock()
    settlement_processor = MagicMock()
    monitor_blocks = []
    processor = ExchangeProcessor(mock_config_trading["exchange_config"], nucleus, strategy, execution_processor, settlement_processor, monitor_blocks)
    with patch("ccxt.async_support.binance", return_value=mock_ccxt_exchange), \
         patch.object(processor.logger, "info") as mock_logger:
        processor.exchange_clients["binance"] = mock_ccxt_exchange
        disponibles = await processor.detectar_disponibles()
        assert len(disponibles) == 1
        assert disponibles[0]["name"] == "binance"
        assert mock_logger.called

@pytest.mark.asyncio
async def test_exchange_processor_monitor_exchange(mock_config_trading, nucleus, mock_redis, mock_market_data):
    """Prueba el monitoreo de un exchange."""
    strategy = MomentumStrategy(1000, MagicMock(), MagicMock())
    execution_processor = ExecutionProcessor({"open_trades": {}, "num_exchanges": 1, "capital": 1000}, mock_redis)
    settlement_processor = MagicMock()
    monitor_blocks = [MagicMock(procesar=AsyncMock(return_value={"status": "success"}))]
    processor = ExchangeProcessor(mock_config_trading["exchange_config"], nucleus, strategy, execution_processor, settlement_processor, monitor_blocks)
    processor.analyzer_processor = MagicMock(analizar_volatilidad=AsyncMock(return_value={
        "status": "ok",
        "datos": {"prioritized_pairs": [("BTC/USDT", 0.03)], "avg_volatility": 0.03}
    }))
    processor.trading_pairs_by_exchange = {"binance": ["BTC/USDT"]}
    processor.exchanges = ["binance"]
    processor.task_queues["binance"] = [(asyncio.get_event_loop().time(), "monitor")]
    with patch("psutil.cpu_percent", return_value=50), \
         patch("psutil.virtual_memory", return_value=MagicMock(percent=50)), \
         patch.object(mock_redis, "get", side_effect=lambda key: json.dumps(mock_market_data) if key == "market_data" else None), \
         patch.object(mock_redis, "set", new=AsyncMock()), \
         patch.object(processor.logger, "info") as mock_logger:
        await asyncio.sleep(0.1)  # Permitir que el bucle empiece
        await processor.monitor_exchange("binance", ["BTC/USDT"])
        assert mock_logger.called

# --- Pruebas de ExecutionProcessor ---

@pytest.mark.asyncio
async def test_execution_processor_ejecutar_operacion(mock_config_trading, mock_redis, mock_market_data):
    """Prueba la ejecución de una operación."""
    processor = ExecutionProcessor({
        "open_trades": {},
        "num_exchanges": 1,
        "capital": 1000,
        "cb_max_failures": 3,
        "cb_reset_timeout": 900
    }, mock_redis)
    with patch.object(mock_redis, "get", return_value=json.dumps(mock_market_data)), \
         patch.object(processor.logger, "info") as mock_logger:
        params = {"precio": 50000, "cantidad": 0.1, "activo": "BTC/USDT", "tipo": "buy"}
        async for result in processor.ejecutar_operacion("binance", params, paper_mode=True, trade_multiplier=1):
            assert result["status"] == "success"
            assert "orden_id" in result["result"]
        assert mock_logger.called

@pytest.mark.asyncio
async def test_execution_processor_slippage_exceed(mock_config_trading, mock_redis, mock_market_data):
    """Prueba la ejecución con deslizamiento excesivo."""
    processor = ExecutionProcessor({
        "open_trades": {},
        "num_exchanges": 1,
        "capital": 1000,
        "cb_max_failures": 3,
        "cb_reset_timeout": 900
    }, mock_redis)
    with patch.object(mock_redis, "get", return_value=json.dumps(mock_market_data)), \
         patch.object(processor, "calculate_slippage", return_value=0.02), \
         patch.object(processor.logger, "warning") as mock_logger:
        params = {"precio": 50000, "cantidad": 0.1, "activo": "BTC/USDT", "tipo": "buy"}
        async for result in processor.ejecutar_operacion("binance", params, paper_mode=True, trade_multiplier=1):
            assert result["status"] == "error"
            assert "Deslizamiento excede tolerancia" in result["motivo"]
        assert mock_logger.called

# --- Pruebas de MacroProcessor ---

@pytest.mark.asyncio
async def test_macro_processor_fetch_and_publish_data(mock_config_trading, mock_redis):
    """Prueba la obtención y publicación de datos macroeconómicos."""
    processor = MacroProcessor(mock_config_trading["macro_config"], mock_redis)
    with patch.object(processor.alpha_vantage, "fetch_macro_data", return_value={"sp500": 0.02, "nasdaq": 0.01}), \
         patch.object(processor.coinmarketcap, "fetch_crypto_data", return_value={"volume": 1000000, "market_cap": 1000000000}), \
         patch.object(mock_redis, "set", new=AsyncMock()), \
         patch.object(processor.logger, "info") as mock_logger:
        result = await processor.fetch_and_publish_data()
        assert result["status"] == "ok"
        assert "macro" in result["datos"]
        assert mock_logger.called

@pytest.mark.asyncio
async def test_macro_processor_fetch_critical_news(mock_config_trading, mock_redis):
    """Prueba la obtención de noticias críticas."""
    processor = MacroProcessor(mock_config_trading["macro_config"], mock_redis)
    with patch("random.choice", return_value=0.25), \
         patch.object(mock_redis, "set", new=AsyncMock()), \
         patch.object(processor.logger, "info") as mock_logger:
        result = await processor.fetch_critical_news()
        assert result["fed_rate_change"] == 0.25
        assert processor.critical_news
        assert mock_logger.called

# --- Pruebas de IAAnalisisProcessor ---

@pytest.mark.asyncio
async def test_ia_analysis_processor_procesar_datos(mock_config_trading, mock_redis):
    """Prueba el procesamiento de datos con el modelo IA."""
    processor = IAAnalisisProcessor(mock_config_trading, mock_redis)
    with patch("torch.load", side_effect=FileNotFoundError), \
         patch.object(processor.model, "eval"), \
         patch.object(processor.logger, "info") as mock_logger:
        await processor.inicializar()
        inputs = [1000000, 0.01, 0.01, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = await processor.procesar_datos(inputs)
        assert "prediccion" in result
        assert "confianza" in result
        assert mock_logger.called

# --- Pruebas de PredictorProcessor ---

@pytest.mark.asyncio
async def test_predictor_processor_predecir_tendencias(mock_config_trading, mock_redis, mock_market_data):
    """Prueba la predicción de tendencias."""
    processor = PredictorProcessor(mock_config_trading["predictor_config"], mock_redis)
    with patch.object(mock_redis, "get", return_value=json.dumps(mock_market_data)), \
         patch.object(mock_redis, "set", new=AsyncMock()), \
         patch.object(processor.logger, "info") as mock_logger:
        result = await processor.predecir_tendencias()
        assert result["status"] == "ok"
        assert "predicciones" in result
        assert mock_logger.called

# --- Pruebas de SettlementProcessor ---

@pytest.mark.asyncio
async def test_settlement_processor_daily_close(mock_config_trading, mock_redis, mock_market_data):
    """Prueba el proceso de cierre diario."""
    strategy = MagicMock()
    execution_processor = ExecutionProcessor({"open_trades": {}, "num_exchanges": 1, "capital": 1000}, mock_redis)
    capital_processor = CapitalProcessor(mock_config_trading["capital_config"], mock_redis, MagicMock(), MagicMock())
    processor = SettlementProcessor(mock_config_trading["settlement_config"], mock_redis, MagicMock(), MagicMock(), strategy, execution_processor, capital_processor)
    processor.open_trades = {"binance:BTC/USDT": {"orden_id": "test_order", "precio": 50000, "cantidad": 0.1, "tipo": "buy", "timestamp": "2025-04-22T12:00:00", "close_timestamp": "2025-04-22T12:00:00"}}
    with patch.object(processor.trading_db, "save_order", new=AsyncMock()), \
         patch.object(processor.trading_db, "save_report", new=AsyncMock()), \
         patch.object(mock_redis, "set", new=AsyncMock()), \
         patch.object(mock_redis, "xadd", new=AsyncMock()), \
         patch.object(processor, "backup_state", new=AsyncMock()), \
         patch.object(processor.logger, "info") as mock_logger:
        await processor.daily_close_process()
        assert len(processor.open_trades) == 0
        assert mock_logger.called

# --- Pruebas de CapitalProcessor ---

@pytest.mark.asyncio
async def test_capital_processor_distribuir_capital(mock_config_trading, mock_redis):
    """Prueba la distribución de capital entre exchanges."""
    processor = CapitalProcessor(mock_config_trading["capital_config"], mock_redis, MagicMock(), MagicMock())
    with patch.object(processor.logger, "info") as mock_logger:
        result = processor.distribuir_capital(["binance", "kucoin"])
        assert "binance" in result
        assert "kucoin" in result
        assert result["binance"] == 350.0  # 1000 * 0.7 / 2
        assert mock_logger.called

# --- Pruebas de MonitorProcessor ---

@pytest.mark.asyncio
async def test_monitor_processor_analizar_volatilidad(mock_config_trading, mock_redis):
    """Prueba el análisis de volatilidad."""
    processor = MonitorProcessor(mock_config_trading["monitor_config"], mock_redis)
    with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"priceChangePercent": "3.0"})
        mock_get.return_value.__aenter__.return_value = mock_response
        with patch.object(mock_redis, "set", new=AsyncMock()), \
             patch.object(processor.logger, "info") as mock_logger:
            result = await processor.analizar_volatilidad()
            assert result["status"] == "ok"
            assert len(result["datos"]) == 1
            assert mock_logger.called

# --- Pruebas de MomentumStrategy ---

@pytest.mark.asyncio
async def test_momentum_strategy_calculate_momentum(mock_config_trading, mock_redis, mock_market_data):
    """Prueba el cálculo de momentum."""
    ia_processor = MagicMock(procesar_datos=AsyncMock(return_value={"prediccion": 0.6}))
    predictor_processor = MagicMock(predecir_tendencias=AsyncMock(return_value={"predicciones": [{"symbol": "BTC/USDT", "tendencia": "alcista"}]}))
    strategy = MomentumStrategy(1000, ia_processor, predictor_processor)
    with patch.object(strategy.logger, "info") as mock_logger:
        prices = [50000 + i * 100 for i in range(50)]
        sentiment = await strategy.calculate_momentum(mock_market_data["macro"], mock_market_data["crypto"]["BTC/USDT"], prices, 0.01)
        assert sentiment > 0
        assert mock_logger.called

# --- Pruebas de TradingDB ---

@pytest.mark.asyncio
async def test_trading_db_connect(mock_config_trading, mock_db_pool):
    """Prueba la conexión a la base de datos."""
    db = TradingDB(mock_config_trading["db_config"])
    with patch("asyncpg.create_pool", return_value=mock_db_pool), \
         patch.object(db.logger, "info") as mock_logger:
        await db.connect()
        assert db.pool == mock_db_pool
        assert mock_logger.called

# --- Pruebas de CircuitBreaker ---

@pytest.mark.asyncio
async def test_circuit_breaker():
    """Prueba el comportamiento del CircuitBreaker."""
    cb = CircuitBreaker(max_failures=2, reset_timeout=1)
    with patch.object(cb.logger, "info") as mock_logger_info, \
         patch.object(cb.logger, "warning") as mock_logger_warning:
        assert cb.check()  # Debería permitir la primera llamada
        cb.register_failure()
        cb.register_failure()  # Activa el breaker después de 2 fallos
        assert not cb.check()  # Debería estar activo
        assert mock_logger_warning.called
        await asyncio.sleep(1.1)  # Esperar el reset_timeout
        assert cb.check()  # Debería haberse reseteado
        assert mock_logger_info.called

# --- Pruebas de Integración ---

@pytest.mark.asyncio
async def test_crypto_trading_integration(mock_crypto_trading, mock_redis, mock_market_data):
    """Prueba de integración completa del plugin."""
    with patch.object(mock_redis, "get", return_value=json.dumps(mock_market_data)), \
         patch.object(mock_redis, "set", new=AsyncMock()), \
         patch.object(mock_redis, "xadd", new=AsyncMock()), \
         patch.object(mock_crypto_trading.logger, "info") as mock_logger:
        await mock_crypto_trading.inicializar(mock_crypto_trading.nucleus)
        result = await mock_crypto_trading.manejar_comando({"action": "ejecutar_operacion", "params": {"exchange": "binance", "pair": "BTC/USDT", "side": "buy"}})
        assert result["status"] == "success"
        assert mock_logger.called

# --- Pruebas de Manejo de Caídas del Mercado ---

@pytest.mark.asyncio
async def test_settlement_processor_handle_market_crash(mock_config_trading, mock_redis, mock_market_data):
    """Prueba el manejo de caídas del mercado."""
    mock_market_data["macro"]["sp500"] = -0.3  # Simula una caída del mercado
    strategy = MagicMock()
    execution_processor = ExecutionProcessor({"open_trades": {}, "num_exchanges": 1, "capital": 1000}, mock_redis)
    capital_processor = CapitalProcessor(mock_config_trading["capital_config"], mock_redis, MagicMock(), MagicMock())
    processor = SettlementProcessor(mock_config_trading["settlement_config"], mock_redis, MagicMock(), MagicMock(), strategy, execution_processor, capital_processor)
    with patch.object(mock_redis, "get", return_value=json.dumps(mock_market_data)), \
         patch.object(mock_redis, "xadd", new=AsyncMock()), \
         patch.object(processor.nucleus, "publicar_alerta", new=AsyncMock()), \
         patch.object(processor.logger, "warning") as mock_logger:
        await processor.handle_market_crash()
        assert processor.is_paused
        assert mock_logger.called
