#!/usr/bin/env python3
# tests/plugins/test_user_management.py
"""
test_user_management.py
Pruebas unitarias para el plugin user_management con soporte para roles.
"""

import pytest
import asyncio
import json
import zstandard as zstd
from src.plugins.user_management.processors.user_processor import UserProcessor
from src.utils.config import load_secrets
from corec.entidad_base import Event

@pytest.mark.asyncio
async def test_user_processor(monkeypatch):
    async def mock_connect(self):
        return True

    async def mock_save_user(self, **kwargs):
        pass

    async def mock_update_user(self, **kwargs):
        pass

    async def mock_delete_user(self, user_id):
        pass

    async def mock_get_user(self, user_id):
        return {"user_id": user_id, "role": "superadmin", "notification_preferences": {"email": True}}

    async def mock_xadd(self, stream, data):
        pass

    monkeypatch.setattr("src.plugins.user_management.utils.db.UserDB.connect", mock_connect)
    monkeypatch.setattr("src.plugins.user_management.utils.db.UserDB.save_user", mock_save_user)
    monkeypatch.setattr("src.plugins.user_management.utils.db.UserDB.update_user", mock_update_user)
    monkeypatch.setattr("src.plugins.user_management.utils.db.UserDB.delete_user", mock_delete_user)
    monkeypatch.setattr("src.plugins.user_management.utils.db.UserDB.get_user", mock_get_user)
    monkeypatch.setattr("redis.asyncio.Redis.xadd", mock_xadd)

    config = load_secrets("configs/plugins/user_management/user_management.yaml")
    processor = UserProcessor(config, None, config.get("db_config"))
    await processor.inicializar(None)

    # Probar registro
    event = Event(
        canal="user_data",
        datos={"data": zstd.compress(json.dumps({"action": "register", "user_id": "test_user", "email": "test@example.com", "password": "pass", "name": "Test", "role": "admin", "notification_preferences": {"email": True}, "requester_id": "superadmin"}).encode())},
        destino="user_management"
    )
    await processor.manejar_evento(event)
    assert await processor.check_permission("test_user", "manage_users")

    # Probar actualización
    event = Event(
        canal="user_data",
        datos={"data": zstd.compress(json.dumps({"action": "update", "user_id": "test_user", "role": "superadmin", "requester_id": "superadmin"}).encode())},
        destino="user_management"
    )
    await processor.manejar_evento(event)

    # Probar eliminación
    event = Event(
        canal="user_data",
        datos={"data": zstd.compress(json.dumps({"action": "delete", "user_id": "test_user", "requester_id": "superadmin"}).encode())},
        destino="user_management"
    )
    await processor.manejar_evento(event)

    await processor.detener()