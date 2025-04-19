#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugins/crypto_trading/utils/db.py
Gestión de trading_db para el plugin CryptoTrading.
"""
import psycopg2
import logging
import json
from typing import Dict, Any

class TradingDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
        self.logger = logging.getLogger("TradingDB")

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            cur = self.conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    exchange TEXT,
                    symbol TEXT,
                    market TEXT,
                    price REAL,
                    timestamp REAL
                ) PARTITION BY RANGE (timestamp)
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS market_data_2025_04 PARTITION OF market_data
                FOR VALUES FROM (UNIX_TIMESTAMP('2025-04-01')) TO (UNIX_TIMESTAMP('2025-05-01'))
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timestamp ON market_data (symbol, timestamp DESC)")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    exchange TEXT,
                    order_id TEXT,
                    symbol TEXT,
                    market TEXT,
                    status TEXT,
                    timestamp REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS capital_pool (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    amount REAL,
                    action TEXT,
                    timestamp REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS pool_state (
                    id SERIAL PRIMARY KEY,
                    pool_total REAL,
                    active_capital REAL,
                    timestamp REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS settlement_reports (
                    id SERIAL PRIMARY KEY,
                    date TEXT,
                    total_profit REAL,
                    roi_percent REAL,
                    total_trades INTEGER,
                    report_data JSONB,
                    timestamp REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS macro_data (
                    id SERIAL PRIMARY KEY,
                    data JSONB,
                    timestamp REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    nano_id TEXT,
                    symbol TEXT,
                    prediction JSONB,
                    actual_value REAL,
                    error REAL,
                    macro_context JSONB,
                    timestamp REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prediction_metrics (
                    id SERIAL PRIMARY KEY,
                    nano_id TEXT,
                    mse REAL,
                    mae REAL,
                    predictions_count INTEGER,
                    timestamp REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS system_insights (
                    id SERIAL PRIMARY KEY,
                    timestamp REAL,
                    metrics JSONB,
                    recommendations JSONB,
                    analysis TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT UNIQUE,
                    email TEXT,
                    password TEXT,
                    name TEXT,
                    role TEXT,
                    notification_preferences JSONB,
                    created_at REAL
                )
            """)
            self.conn.commit()
            cur.close()
            self.logger.info("Conectado a trading_db")
            return True
        except Exception as e:
            self.logger.error(f"Error conectando a trading_db: {e}")
            return False

    async def save_price(self, exchange: str, symbol: str, market: str, price: float, timestamp: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO market_data (exchange, symbol, market, price, timestamp) VALUES (%s, %s, %s, %s, %s)",
                (exchange, symbol, market, price, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando precio: {e}")

    async def save_order(self, exchange: str, order_id: str, symbol: str, market: str, status: str, timestamp: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO orders (exchange, order_id, symbol, market, status, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
                (exchange, order_id, symbol, market, status, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando orden: {e}")

    async def save_contribution(self, user_id: str, amount: float, timestamp: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO capital_pool (user_id, amount, action, timestamp) VALUES (%s, %s, %s, %s)",
                (user_id, amount, "contribution", timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando contribución: {e}")

    async def save_withdrawal(self, user_id: str, amount: float, timestamp: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO capital_pool (user_id, amount, action, timestamp) VALUES (%s, %s, %s, %s)",
                (user_id, amount, "withdrawal", timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando retiro: {e}")

    async def get_pool_total(self) -> float:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT COALESCE(SUM(CASE WHEN action = 'contribution' THEN amount ELSE -amount END), 0) FROM capital_pool")
            result = cur.fetchone()[0]
            cur.close()
            return float(result)
        except Exception as e:
            self.logger.error(f"Error obteniendo pool total: {e}")
            return 0.0

    async def get_users(self) -> Dict[str, float]:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT user_id, SUM(CASE WHEN action = 'contribution' THEN amount ELSE -amount END) FROM capital_pool GROUP BY user_id HAVING SUM(CASE WHEN action = 'contribution' THEN amount ELSE -amount END) > 0")
            users = {row[0]: float(row[1]) for row in cur.fetchall()}
            cur.close()
            return users
        except Exception as e:
            self.logger.error(f"Error obteniendo usuarios: {e}")
            return {}

    async def get_active_capital(self) -> float:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT active_capital FROM pool_state ORDER BY timestamp DESC LIMIT 1")
            result = cur.fetchone()
            cur.close()
            return float(result[0]) if result else 0.0
        except Exception as e:
            self.logger.error(f"Error obteniendo capital activo: {e}")
            return 0.0

    async def update_pool(self, pool_total: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO pool_state (pool_total, active_capital, timestamp) VALUES (%s, %s, %s)",
                (pool_total, await self.get_active_capital(), datetime.utcnow().timestamp())
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error actualizando pool: {e}")

    async def update_active_capital(self, active_capital: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO pool_state (pool_total, active_capital, timestamp) VALUES (%s, %s, %s)",
                (await self.get_pool_total(), active_capital, datetime.utcnow().timestamp())
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error actualizando capital activo: {e}")

    async def save_report(self, date: str, total_profit: float, roi_percent: float, total_trades: int, report_data: Dict[str, Any], timestamp: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO settlement_reports (date, total_profit, roi_percent, total_trades, report_data, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
                (date, total_profit, roi_percent, total_trades, json.dumps(report_data), timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando reporte: {e}")

    async def save_macro_data(self, data: Dict[str, Any]):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO macro_data (data, timestamp) VALUES (%s, %s)",
                (json.dumps(data), data.get("timestamp", datetime.utcnow().timestamp()))
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando datos macro: {e}")

    async def save_prediction(self, nano_id: str, symbol: str, prediction: Dict[str, Any], actual_value: float, error: float, macro_context: Dict[str, Any], timestamp: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO predictions (nano_id, symbol, prediction, actual_value, error, macro_context, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (nano_id, symbol, json.dumps(prediction), actual_value, error, json.dumps(macro_context), timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando predicción: {e}")

    async def save_metrics(self, nano_id: str, mse: float, mae: float, predictions_count: int, timestamp: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO prediction_metrics (nano_id, mse, mae, predictions_count, timestamp) VALUES (%s, %s, %s, %s, %s)",
                (nano_id, mse, mae, predictions_count, timestamp)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando métricas: {e}")

    async def save_insight(self, timestamp: float, metrics: Dict[str, Any], recommendations: List[Dict[str, Any]], analysis: str):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO system_insights (timestamp, metrics, recommendations, analysis) VALUES (%s, %s, %s, %s)",
                (timestamp, json.dumps(metrics), json.dumps(recommendations), analysis)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando insight: {e}")

    async def save_user(self, user_id: str, email: str, password: str, name: str, role: str, notification_preferences: Dict[str, Any], created_at: float):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO users (user_id, email, password, name, role, notification_preferences, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (user_id, email, password, name, role, json.dumps(notification_preferences), created_at)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando usuario: {e}")

    async def get_user(self, user_id: str) -> Dict[str, Any]:
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT user_id, email, password, name, role, notification_preferences, created_at FROM users WHERE user_id = %s", (user_id,))
            result = cur.fetchone()
            cur.close()
            if result:
                return {
                    "user_id": result[0],
                    "email": result[1],
                    "password": result[2],
                    "name": result[3],
                    "role": result[4],
                    "notification_preferences": json.loads(result[5]),
                    "created_at": result[6]
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error obteniendo usuario: {e}")
            return {}

    async def update_user(self, user_id: str, email: str = None, password: str = None, name: str = None, role: str = None, notification_preferences: Dict[str, Any] = None):
        try:
            cur = self.conn.cursor()
            updates = []
            values = []
            if email:
                updates.append("email = %s")
                values.append(email)
            if password:
                updates.append("password = %s")
                values.append(password)
            if name:
                updates.append("name = %s")
                values.append(name)
            if role:
                updates.append("role = %s")
                values.append(role)
            if notification_preferences:
                updates.append("notification_preferences = %s")
                values.append(json.dumps(notification_preferences))
            if updates:
                values.append(user_id)
                query = f"UPDATE users SET {', '.join(updates)} WHERE user_id = %s"
                cur.execute(query, values)
                self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error actualizando usuario: {e}")

    async def update_user_preferences(self, user_id: str, notification_preferences: Dict[str, Any]):
        try:
            cur = self.conn.cursor()
            cur.execute(
                "UPDATE users SET notification_preferences = %s WHERE user_id = %s",
                (json.dumps(notification_preferences), user_id)
            )
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error actualizando preferencias: {e}")

    async def delete_user(self, user_id: str):
        try:
            cur = self.conn.cursor()
            cur.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error eliminando usuario: {e}")

    async def disconnect(self):
        if self.conn:
            self.conn.close()
            self.logger.info("Desconectado de trading_db")