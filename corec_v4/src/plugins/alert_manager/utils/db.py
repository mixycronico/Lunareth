#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/alert_manager/utils/db.py
"""
db.py
Gestión de la base de datos propia para el plugin alert_manager.
"""

import psycopg2
import asyncio
from typing import Dict, Any

class AlertDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            await asyncio.sleep(0)  # Simula operación asíncrona
            return True
        except Exception as e:
            print(f"Error conectando a alert_db: {e}")
            return False

    async def save_alert(self, severity: str, message: str, channel: str, event_data: Dict[str, Any], timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO alerts (severity, message, channel, event_data, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (severity, message, channel, json.dumps(event_data), timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando alerta: {e}")

    async def disconnect(self) -> None:
        if self.conn:
            self.conn.close()