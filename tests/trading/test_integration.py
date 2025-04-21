import pytest
import asyncio
import json
from plugins.crypto_trading.processors.orchestrator_processor import OrchestratorProcessor

@pytest.mark.asyncio
async def test_full_trading_flow(orchestrator, mock_redis):
    # Simular datos de mercado
    mock_redis.get.side_effect = [
        json.dumps({
            "macro": {"sp500": 0.01},
            "crypto": {
                "BTC/USDT": {"volume": 3000000, "volatility": 0.02, "price_change": 0.01},
                "ETH/USDT": {"volume": 1500000, "volatility": 0.03, "price_change": 0.01}
            }
        }),
        json.dumps({"interval_factor": 1.0, "trade_multiplier": 2, "pause": False}),
        json.dumps({"trend": "alcista", "magnitude": 0.02})
    ]

    # Simular 1 día de trading
    await asyncio.sleep(1)  # Dar tiempo para que el flujo se ejecute

    # Verificar que se generaron operaciones
    assert len(orchestrator.open_trades) > 0

    # Simular caída del mercado
    mock_redis.get.side_effect = [
        json.dumps({
            "macro": {"sp500": -0.25},
            "crypto": {
                "BTC/USDT": {"volume": 3000000, "volatility": 0.05, "price_change": -0.3},
                "ETH/USDT": {"volume": 1500000, "volatility": 0.06, "price_change": -0.3}
            }
        })
    ]
    await asyncio.sleep(1)  # Dar tiempo para que se detecte la caída

    # Verificar que las operaciones se cerraron
    assert len(orchestrator.open_trades) == 0
