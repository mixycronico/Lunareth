# tests/test_comm_system.py

import sys, os
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

# Añade la raíz del proyecto para que 'plugins/comm_system' sea importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import corec.core as corec_core
from aioredis import Redis

from plugins.comm_system.processors.memory import ChatMemory
from plugins.comm_system.processors.ai_chat import AIChat
from plugins.comm_system.processors.manager import CommManager

@pytest.mark.asyncio
async def test_chat_memory_load_save(tmp_path):
    history = [("q1", "a1"), ("q2", "a2")]
    blob = corec_core.zstd.compress(json.dumps(history).encode())

    fake_redis = AsyncMock(spec=Redis)
    fake_redis.get.return_value = blob

    mem_file = tmp_path / "mem.json"
    cm = ChatMemory(redis=fake_redis, ttl=10, disk_path=str(mem_file))

    # load()
    loaded = await cm.load()
    assert loaded == history

    # to_openai()
    msgs = cm.to_openai(history, max_pairs=1)
    assert msgs == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]

    # save()
    await cm.save(history)
    fake_redis.set.assert_called_once()
    assert json.loads(mem_file.read_text()) == history

    # add_pair()
    fake_redis.get.return_value = None
    cm2 = ChatMemory(redis=fake_redis, ttl=10, disk_path=str(tmp_path / "mem2.json"))
    await cm2.add_pair("new_q", "new_a")
    assert fake_redis.set.call_count >= 1

@pytest.mark.asyncio
async def test_ai_chat_calls_openrouter(monkeypatch):
    config = {
        "openrouter_model":   "mymodel",
        "openrouter_api_key": "mykey",
        "openrouter_api_base":"https://openrouter.ai/api/v1",
        "max_tokens":         5,
        "temperature":        0.2,
    }
    chat = AIChat(config)

    fake_resp = MagicMock()
    fake_resp.choices = [MagicMock(message=MagicMock(content="  respuesta  "))]
    monkeypatch.setattr("openai.ChatCompletion.create", lambda **kw: fake_resp)

    out = await chat.chat([{"role":"user","content":"hola"}], "¿Cómo estás?")
    assert out == "respuesta"

@pytest.mark.asyncio
async def test_manager_handle_chat_and_status():
    fake_nucleus = MagicMock()
    fake_nucleus.redis_config = {"username":"u","password":"p","host":"h","port":6379}
    fake_nucleus.modulos     = {"mod1": object()}
    fake_nucleus.plugins     = {"plug1": object()}

    config = {
        "stream_in":        "in",
        "stream_out":       "out",
        "openrouter_model": "m",
        "openrouter_api_key":"k",
        "openrouter_api_base":"b",
        "max_tokens":       10,
        "temperature":      0.1,
        "memory_ttl":       60,
        "memory_file":      "memory.json"
    }
    mgr = CommManager(fake_nucleus, config)

    mgr.redis   = AsyncMock()
    mgr.memory  = AsyncMock()
    mgr.chat_ai = AsyncMock()

    mgr.memory.load.return_value    = [("q","a")]
    mgr.memory.to_openai.return_value = [
        {"role":"user","content":"q"},
        {"role":"assistant","content":"a"}
    ]
    mgr.chat_ai.chat.return_value   = "respuesta IA"

    # chat
    resp = await mgr.handle_command({"action":"chat","params":{"mensaje":"Prueba"}})
    mgr.memory.add_pair.assert_awaited_with("Prueba","respuesta IA")
    assert resp == {"status":"ok","texto":"respuesta IA"}

    # status
    resp = await mgr.handle_command({"action":"status"})
    assert resp["status"] == "ok"
    assert resp["modulos"] == ["mod1"]
    assert resp["plugins"] == ["plug1"]

    # unknown
    resp = await mgr.handle_command({"action":"foo"})
    assert resp["status"] == "error"
    assert "desconocida" in resp["message"]