#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/cli_manager/utils/db.py
"""
db.py
Gestión de la base de datos para el plugin cli_manager.
"""

import psycopg2
import asyncio
import json
from typing import Dict, Any

class CLIDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            await asyncio.sleep(0)
            return True
        except Exception as e:
            print(f"Error conectando a cli_db: {e}")
            return False

    async def save_action(self, action: str, user_id: str, timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO actions (action, user_id, timestamp) VALUES (%s, %s, %s)",
                (action, user_id, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando acción: {e}")

    async def save_goal(self, goal_id: str, goal: Dict[str, Any], user_id: str, timestamp: float) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO goals (goal_id, goal_data, user_id, timestamp) VALUES (%s, %s, %s, %s)",
                (goal_id, json.dumps(goal), user_id, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando meta: {e}")

    async def get_goals(self) -> Dict[str, Dict[str, Any]]:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT goal_id, goal_data FROM goals")
            goals = {row[0]: json.loads(row[1]) for row in cur.fetchall()}
            cur.close()
            return goals
        except Exception as e:
            print(f"Error obteniendo metas: {e}")
            return {}

    async def disconnect(self) -> None:
        if self.conn:
            self.conn.close()