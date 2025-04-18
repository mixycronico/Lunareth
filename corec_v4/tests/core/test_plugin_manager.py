import pytest
import asyncio
from src.plugins.plugin_manager import PluginManager
from src.core.nucleus import CoreCNucleus

@pytest.mark.asyncio
async def test_plugin_manager_carga_vacia():
    manager = PluginManager()
    nucleus = CoreCNucleus(instance_id="test_corec1")
    await manager.cargar_plugins(nucleus)
    assert len(manager.plugins) == 0
    await manager.detener_plugins()