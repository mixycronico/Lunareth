#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/trading_execution/utils/db.py
"""
db.py
Gestión de la base de datos propia para el plugin trading_execution.
"""

import psycopg2
import asyncio
from typing import Dict, Any

class ExecutionDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            await asyncio.sleep(0)  # Simula operación asíncrona
            return True
        except Exception as e:
            print(f"Error conectando a execution_db: {e}")
            return False

    async def save_order(self, exchange: str, order_id: str, symbol: str, market: str, side: str, quantity: float, price: float, status: str, timestamp: float, close_reason: str = None, close_timestamp: float = None) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO orders (exchange, order_id, symbol, market, side, quantity, price, status, timestamp, close_reason, close_timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (exchange, order_id, symbol, market, side, quantity, price, status, timestamp, close_reason, close_timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando orden {order_id} en {exchange}: {e}")

    async def disconnect(self) -> None:
        if self.conn:
            self.conn.close()