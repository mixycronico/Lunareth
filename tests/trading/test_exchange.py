import pytest
import asyncio
import json
from plugins.crypto_trading.processors.exchange_processor import ExchangeProcessor

@pytest.mark.asyncio
async def test_exchange_obtener_top_activos(mock_nucleus, mock_redis, mock_trading_db, config):
    processor = ExchangeProcessor(config, mock_nucleus, None, None, None, [])
    client = AsyncMock()
    client.fetch_tickers.return_value = {
        "BTC/USDT": {"quoteVolume": 3000000},
        "ETH/USDT": {"quoteVolume": 1500000},
        "SOL/USDT": {"quoteVolume": 500000},
        "ADA/USDT": {"quoteVolume": 400000}
    }
    activos = await processor.obtener_top_activos(client)
    assert "BTC/USDT" in activos
    assert "ETH/USDT" in activos
    assert "SOL/USDT" in activos
    assert "ADA/USDT" in activos
    assert len(activos) <= 12

@pytest.mark.asyncio
async def test_exchange_monitor_with_slippage(mock_nucleus, mock_redis, mock_trading_db, config):
    processor = ExchangeProcessor(config, mock_nucleus, AsyncMock(), AsyncMock(), AsyncMock(), [])
    pairs = ["BTC/USDT", "ETH/USDT"]
    processor.strategy = AsyncMock()
    processor.strategy.calculate_momentum.return_value = 410.50
    processor.strategy.decide_trade.return_value = "buy"
    processor.strategy.get_trade_multiplier.return_value = 1
    processor.execution_processor = AsyncMock()
    processor.execution_processor.ejecutar_operacion.return_value = AsyncMock()
    processor.settlement_processor = AsyncMock()
    processor.settlement_processor.update_capital_after_trade = AsyncMock()
    processor.monitor_blocks = [AsyncMock()]
    processor.monitor_blocks[0].procesar.return_value = {"status": "success"}

    mock_redis.get.side_effect = [
        json.dumps({"crypto": {"BTC/USDT": {"volume": 3000000, "volatility": 0.02}}}),
        json.dumps({"interval_factor": 1.0, "trade_multiplier": 2, "pause": False})
    ]

    await processor.monitor_exchange("binance", pairs, initial_offset=0)
    processor.execution_processor.ejecutar_operacion.assert_called()
