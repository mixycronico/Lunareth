import pytest
import asyncio
import json
from plugins.crypto_trading.processors.execution_processor import ExecutionProcessor

@pytest.mark.asyncio
async def test_execution_with_slippage(mock_redis):
    config = {
        "capital": 1000,
        "open_trades": {},
        "num_exchanges": 2
    }
    redis_mock = mock_redis
    redis_mock.get.return_value = json.dumps({
        "crypto": {
            "SOL/USDT": {"volume": 500000, "volatility": 0.06}
        }
    })
    processor = ExecutionProcessor(config, redis_mock)

    params = {
        "precio": 31.50,
        "cantidad": 2,
        "activo": "SOL/USDT",
        "tipo": "buy"
    }
    trade_multiplier = 1

    async for result in processor.ejecutar_operacion("binance", params, paper_mode=True, trade_multiplier=trade_multiplier):
        assert result["status"] == "success"
        assert "orden_id" in result["result"]
        assert result["result"]["precio"] > 31.50  # Deslizamiento aplicado
        assert result["result"]["cantidad"] == 2

@pytest.mark.asyncio
async def test_execution_slippage_exceeds_tolerance(mock_redis):
    config = {
        "capital": 1000,
        "open_trades": {},
        "num_exchanges": 2
    }
    redis_mock = mock_redis
    redis_mock.get.return_value = json.dumps({
        "crypto": {
            "SOL/USDT": {"volume": 100000, "volatility": 0.1}  # Alta volatilidad, bajo volumen
        }
    })
    processor = ExecutionProcessor(config, redis_mock)

    params = {
        "precio": 31.50,
        "cantidad": 2,
        "activo": "SOL/USDT",
        "tipo": "buy"
    }
    trade_multiplier = 1

    async for result in processor.ejecutar_operacion("binance", params, paper_mode=True, trade_multiplier=trade_multiplier):
        assert result["status"] == "error"
        assert "Deslizamiento excede tolerancia" in result["motivo"]
