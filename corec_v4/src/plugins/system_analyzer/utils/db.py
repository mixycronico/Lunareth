#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/plugins/system_analyzer/utils/db.py
"""
db.py
Gestión de la base de datos para el plugin system_analyzer.
"""

import psycopg2
import asyncio
import json
from typing import Dict, Any, List

class AnalyzerDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            await asyncio.sleep(0)
            return True
        except Exception as e:
            print(f"Error conectando a analyzer_db: {e}")
            return False

    async def save_insight(self, timestamp: float, metrics: Dict[str, Any], recommendations: List[Dict[str, Any]], analysis: str) -> None:
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO insights (timestamp, metrics, recommendations, analysis) VALUES (%s, %s, %s, %s)",
                (timestamp, json.dumps(metrics), json.dumps(recommendations), analysis)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            print(f"Error guardando insight: {e}")

    async def disconnect(self) -> None:
        if self.conn:
            self.conn.close()