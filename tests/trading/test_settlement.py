import pytest
import asyncio
import json
from plugins.crypto_trading.processors.settlement_processor import SettlementProcessor

@pytest.mark.asyncio
async def test_settlement_slippage_monitoring(mock_redis, mock_trading_db, mock_nucleus):
    config = {"total_capital": 1000}
    execution_processor = AsyncMock()
    execution_processor.slippage_history = [0.012, 0.008, 0.007]  # Deslizamiento promedio > 0.5%
    processor = SettlementProcessor(config, mock_redis, mock_trading_db, mock_nucleus, AsyncMock(), execution_processor, AsyncMock())

    await processor.monitor_slippage()
    assert processor.trade_size_adjustment < 1.0  # Tamaño ajustado a la baja
    assert processor.average_slippage > 0.005  # Promedio de deslizamiento alto

@pytest.mark.asyncio
async def test_settlement_market_crash_with_slippage(mock_redis, mock_trading_db, mock_nucleus):
    config = {"total_capital": 1000}
    execution_processor = AsyncMock()
    processor = SettlementProcessor(config, mock_redis, mock_trading_db, mock_nucleus, AsyncMock(), execution_processor, AsyncMock())
    processor.open_trades = {
        "binance:SOL/USDT": {
            "orden_id": "test_order",
            "precio": 31.72,
            "cantidad": 2,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

    mock_redis.get.return_value = json.dumps({
        "macro": {"sp500": -0.25},
        "crypto": {"SOL/USDT": {"price_change": -0.1}}
    })

    await processor.handle_market_crash()
    assert len(processor.open_trades) == 0  # Operaciones cerradas
    assert processor.is_paused == True  # Pausa activada
