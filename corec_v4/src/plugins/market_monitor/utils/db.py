#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/market_monitor/utils/db.py
"""
db.py
Gestión de la base de datos propia para el plugin market_monitor.
"""

import psycopg2
import asyncio
from typing import Dict, Any

class MonitorDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            await asyncio.sleep(0)  # Simula operación asíncrona
            return True
        except Exception as e:
            print(f"Error conectando a monitor_db: {e}")
            return False

    async def save_price(self, symbol: str, price: float, timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO market_data (symbol, price, timestamp)
                VALUES (%s, %s, %s)
                """,
                (symbol, price, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando precio para {symbol}: {e}")

    async def disconnect(self) -> None:
        if self.conn:
            self.conn.close()