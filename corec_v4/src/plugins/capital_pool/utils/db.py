#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/capital_pool/utils/db.py
"""
db.py
Gestión de la base de datos propia para el plugin capital_pool.
"""

import psycopg2
import asyncio
from typing import Dict, Any

class CapitalDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            await asyncio.sleep(0)  # Simula operación asíncrona
            return True
        except Exception as e:
            print(f"Error conectando a capital_db: {e}")
            return False

    async def save_contribution(self, user_id: str, amount: float, timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO contributions (user_id, amount, timestamp)
                VALUES (%s, %s, %s)
                """,
                (user_id, amount, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando contribución para {user_id}: {e}")

    async def save_withdrawal(self, user_id: str, amount: float, timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO withdrawals (user_id, amount, timestamp)
                VALUES (%s, %s, %s)
                """,
                (user_id, amount, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando retiro para {user_id}: {e}")

    async def update_pool(self, total: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO pool_state (total, timestamp)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET total = %s, timestamp = %s
                """,
                (total, datetime.utcnow().timestamp(), total, datetime.utcnow().timestamp())
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error actualizando pool: {e}")

    async def update_active_capital(self, active: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                """
                INSERT INTO pool_state (active_capital, timestamp)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET active_capital = %s, timestamp = %s
                """,
                (active, datetime.utcnow().timestamp(), active, datetime.utcnow().timestamp())
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error actualizando capital activo: {e}")

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