#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/daily_settlement/utils/db.py
"""
db.py
Gestión de la base de datos propia para el plugin daily_settlement.
"""

import psycopg2
import asyncio
from typing import Dict, Any

class SettlementDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            await asyncio.sleep(0)  # Simula operación asíncrona
            return True
        except Exception as e:
            print(f"Error conectando a settlement_db: {e}")
            return False

    async def save_report(self, date: str, total_profit: float, roi_percent: float, total_trades: int, report_data: Dict[str, Any]) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO reports (date, total_profit, roi_percent, total_trades, report_data, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (date, total_profit, roi_percent, total_trades, json.dumps(report_data), datetime.utcnow().timestamp())
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando reporte {date}: {e}")

    async def get_pool_total(self) -> float:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT total FROM pool_state WHERE id = 1")
            result = cur.fetchone()
            cur.close()
            return result[0] if result else 0.0
        except Exception as e:
            print(f"Error obteniendo pool total: {e}")
            return 0.0

    async def get_active_capital(self) -> float:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT active_capital FROM pool_state WHERE id = 1")
            result = cur.fetchone()
            cur.close()
            return result[0] if result else 0.0
        except Exception as e:
            print(f"Error obteniendo capital activo: {e}")
            return 0.0

    async def get_users(self) -> Dict[str, float]:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT user_id, SUM(amount) FROM contributions GROUP BY user_id")
            users = {row[0]: row[1] for row in cur.fetchall()}
            cur.execute("SELECT user_id, SUM(amount) FROM withdrawals GROUP BY user_id")
            for row in cur.fetchall():
                users[row[0]] = users.get(row[0], 0) - row[1]
            cur.close()
            return {k: v for k, v in users.items() if v > 0}
        except Exception as e:
            print(f"Error obteniendo usuarios: {e}")
            return {}

    async def disconnect(self) -> None:
        if self.conn:
            self.conn.close()