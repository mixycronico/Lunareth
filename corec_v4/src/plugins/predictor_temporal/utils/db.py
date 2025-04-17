#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/predictor_temporal/utils/db.py
"""
db.py
Gestión de la base de datos propia para el plugin predictor_temporal.
"""

import psycopg2
import asyncio
import json
from typing import Dict, Any, Optional

class PredictorDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            await asyncio.sleep(0)  # Simula operación asíncrona
            return True
        except Exception as e:
            print(f"Error conectando a predictor_db: {e}")
            return False

    async def save_prediction(self, nano_id: str, symbol: str, prediction: Any, actual_value: Optional[float], 
                             error: Optional[float], macro_context: Dict, timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO predictions (nano_id, symbol, prediction, actual_value, error, macro_context, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (nano_id, symbol, json.dumps(prediction), actual_value, error, json.dumps(macro_context), timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando predicción: {e}")

    async def save_metrics(self, nano_id: str, mse: float, mae: float, predictions_count: int, timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO metrics (nano_id, mse, mae, predictions_count, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (nano_id, mse, mae, predictions_count, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando métricas: {e}")

    async def disconnect(self) -> None:
        if self.conn:
            self.conn.close()