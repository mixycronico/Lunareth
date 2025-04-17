#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/macro_sync/utils/db.py
"""
db.py
Gestión de la base de datos propia para el plugin macro_sync.
"""

import psycopg2
import asyncio
from typing import Dict, Any

class MacroDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            await asyncio.sleep(0)  # Simula operación asíncrona
            return True
        except Exception as e:
            print(f"Error conectando a macro_db: {e}")
            return False

    async def save_macro_data(self, sp500_price: float, nasdaq_price: float, vix_price: float, 
                             gold_price: float, oil_price: float, altcoins_volume: float, 
                             news_sentiment: float, timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO macro_metrics (sp500_price, nasdaq_price, vix_price, gold_price, oil_price, altcoins_volume, news_sentiment, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (sp500_price, nasdaq_price, vix_price, gold_price, oil_price, altcoins_volume, news_sentiment, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando datos macro: {e}")

    async def disconnect(self) -> None:
        if self.conn:
            self.conn.close()