#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/exchange_sync/utils/db.py
"""
db.py
Gestión de la base de datos propia para el plugin exchange_sync.
"""

import psycopg2
import asyncio
from typing import Dict, Any

class ExchangeDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            await asyncio.sleep(0)  # Simula operación asíncrona
            return True
        except Exception as e:
            print(f"Error conectando a exchange_db: {e}")
            return False

    async def save_price(self, exchange: str, symbol: str, market: str, price: float, timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO exchange_data (exchange, symbol, market, price, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (exchange, symbol, market, price, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando precio para {symbol} en {exchange}: {e}")

    async def save_order(self, exchange: str, order_id: str, symbol: str, market: str, status: str, timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO open_orders (exchange, order_id, symbol, market, status, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (exchange, order_id, symbol, market, status, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando orden {order_id} en {exchange}: {e}")

    async def disconnect(self) -> None:
        if self.conn:
            self.conn.close()