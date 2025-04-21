import logging
import json
import psycopg2
from typing import Dict, Any, List
from datetime import datetime

class TradingDB:
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = None
        self.logger = logging.getLogger("TradingDB")

    async def connect(self) -> bool:
        try:
            self.conn = psycopg2.connect(**self.db_config)
            cur = self.conn.cursor()

            # Estructura de tablas necesarias
            cur.execute("""CREATE TABLE IF NOT EXISTS market_data (
                id SERIAL PRIMARY KEY,
                exchange TEXT,
                symbol TEXT,
                market TEXT,
                price REAL,
                timestamp REAL
            ) PARTITION BY RANGE (timestamp)""")

            cur.execute("""CREATE TABLE IF NOT EXISTS orders (
                id SERIAL PRIMARY KEY,
                exchange TEXT,
                order_id TEXT,
                symbol TEXT,
                market TEXT,
                status TEXT,
                timestamp REAL
            )""")

            cur.execute("""CREATE TABLE IF NOT EXISTS capital_pool (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                amount REAL,
                action TEXT,
                timestamp REAL
            )""")

            cur.execute("""CREATE TABLE IF NOT EXISTS pool_state (
                id SERIAL PRIMARY KEY,
                pool_total REAL,
                active_capital REAL,
                timestamp REAL
            )""")

            cur.execute("""CREATE TABLE IF NOT EXISTS settlement_reports (
                id SERIAL PRIMARY KEY,
                date TEXT,
                total_profit REAL,
                roi_percent REAL,
                total_trades INTEGER,
                report_data JSONB,
                timestamp REAL
            )""")

            cur.execute("""CREATE TABLE IF NOT EXISTS macro_data (
                id SERIAL PRIMARY KEY,
                data JSONB,
                timestamp REAL
            )""")

            cur.execute("""CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                nano_id TEXT,
                symbol TEXT,
                prediction JSONB,
                actual_value REAL,
                error REAL,
                macro_context JSONB,
                timestamp REAL
            )""")

            cur.execute("""CREATE TABLE IF NOT EXISTS prediction_metrics (
                id SERIAL PRIMARY KEY,
                nano_id TEXT,
                mse REAL,
                mae REAL,
                predictions_count INTEGER,
                timestamp REAL
            )""")

            cur.execute("""CREATE TABLE IF NOT EXISTS system_insights (
                id SERIAL PRIMARY KEY,
                timestamp REAL,
                metrics JSONB,
                recommendations JSONB,
                analysis TEXT
            )""")

            self.conn.commit()
            cur.close()
            self.logger.info("Conexión y migraciones completadas en trading_db")
            return True

        except Exception as e:
            self.logger.error(f"Error al conectar con trading_db: {e}")
            return False

    async def disconnect(self):
        if self.conn:
            self.conn.close()
            self.logger.info("Desconectado de trading_db")

    async def _insert(self, query: str, params: tuple, label: str):
        try:
            cur = self.conn.cursor()
            cur.execute(query, params)
            self.conn.commit()
            cur.close()
        except Exception as e:
            self.logger.error(f"Error guardando {label}: {e}")

    async def _query_single(self, query: str, label: str) -> float:
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            result = cur.fetchone()
            cur.close()
            return float(result[0]) if result else 0.0
        except Exception as e:
            self.logger.error(f"Error obteniendo {label}: {e}")
            return 0.0

    async def save_price(self, exchange, symbol, market, price, timestamp):
        await self._insert("INSERT INTO market_data (exchange, symbol, market, price, timestamp) VALUES (%s, %s, %s, %s, %s)",
                           (exchange, symbol, market, price, timestamp), "precio")

    async def save_order(self, exchange, order_id, symbol, market, status, timestamp):
        await self._insert("INSERT INTO orders (exchange, order_id, symbol, market, status, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
                           (exchange, order_id, symbol, market, status, timestamp), "orden")

    async def save_contribution(self, user_id, amount, timestamp):
        await self._insert("INSERT INTO capital_pool (user_id, amount, action, timestamp) VALUES (%s, %s, %s, %s)",
                           (user_id, amount, "contribution", timestamp), "contribución")

    async def save_withdrawal(self, user_id, amount, timestamp):
        await self._insert("INSERT INTO capital_pool (user_id, amount, action, timestamp) VALUES (%s, %s, %s, %s)",
                           (user_id, amount, "withdrawal", timestamp), "retiro")

    async def get_pool_total(self) -> float:
        return await self._query_single("SELECT COALESCE(SUM(CASE WHEN action = 'contribution' THEN amount ELSE -amount END), 0) FROM capital_pool", "pool total")

    async def get_active_capital(self) -> float:
        return await self._query_single("SELECT active_capital FROM pool_state ORDER BY timestamp DESC LIMIT 1", "capital activo")

    async def update_pool(self, pool_total: float):
        await self._insert("INSERT INTO pool_state (pool_total, active_capital, timestamp) VALUES (%s, %s, %s)",
                           (pool_total, await self.get_active_capital(), datetime.datetime.utcnow().timestamp()), "pool state")

    async def update_active_capital(self, active_capital: float):
        await self._insert("INSERT INTO pool_state (pool_total, active_capital, timestamp) VALUES (%s, %s, %s)",
                           (await self.get_pool_total(), active_capital, datetime.datetime.utcnow().timestamp()), "capital activo")

    async def save_report(self, date, total_profit, roi_percent, total_trades, report_data, timestamp):
        await self._insert("INSERT INTO settlement_reports (date, total_profit, roi_percent, total_trades, report_data, timestamp) VALUES (%s, %s, %s, %s, %s, %s)",
                           (date, total_profit, roi_percent, total_trades, json.dumps(report_data), timestamp), "reporte")

    async def save_macro_data(self, data):
        await self._insert("INSERT INTO macro_data (data, timestamp) VALUES (%s, %s)",
                           (json.dumps(data), data.get("timestamp", datetime.datetime.utcnow().timestamp())), "macro")

    async def save_prediction(self, nano_id, symbol, prediction, actual_value, error, macro_context, timestamp):
        await self._insert("INSERT INTO predictions (nano_id, symbol, prediction, actual_value, error, macro_context, timestamp) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                           (nano_id, symbol, json.dumps(prediction), actual_value, error, json.dumps(macro_context), timestamp), "predicción")

    async def save_metrics(self, nano_id, mse, mae, predictions_count, timestamp):
        await self._insert("INSERT INTO prediction_metrics (nano_id, mse, mae, predictions_count, timestamp) VALUES (%s, %s, %s, %s, %s)",
                           (nano_id, mse, mae, predictions_count, timestamp), "métricas")

    async def save_insight(self, timestamp, metrics, recommendations, analysis):
        await self._insert("INSERT INTO system_insights (timestamp, metrics, recommendations, analysis) VALUES (%s, %s, %s, %s)",
                           (timestamp, json.dumps(metrics), json.dumps(recommendations), analysis), "insight")
