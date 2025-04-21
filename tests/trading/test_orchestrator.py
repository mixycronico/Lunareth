import pytest
import asyncio
from plugins.crypto_trading.processors.orchestrator_processor import OrchestratorProcessor

@pytest.mark.asyncio
async def test_orchestrator_initialization(mock_nucleus, mock_redis, mock_trading_db, config):
    orchestrator = OrchestratorProcessor(config, mock_nucleus, mock_redis)
    await orchestrator.initialize()
    assert orchestrator.analyzer_processor is not None
    assert orchestrator.capital_processor is not None
    assert orchestrator.exchange_processor is not None
    assert len(orchestrator.trading_blocks) > 0
    assert len(orchestrator.monitor_blocks) > 0
