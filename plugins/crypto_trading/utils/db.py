import asyncpg
import logging

class TradingDB:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("TradingDB")
        self.pool = None

    async def connect(self):
        try:
            self.pool = await asyncpg.create_pool(
                database=self.config["dbname"],
                user=self.config["user"],
                password=self.config["password"],
                host=self.config["host"],
                port=self.config["port"]
            )
            self.logger.info("Conexión a la base de datos establecida")

            # Crear índices para mejorar el rendimiento de consultas frecuentes
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_orders_timestamp ON orders (timestamp);
                    CREATE INDEX IF NOT EXISTS idx_reports_date ON reports (date);
                """)
                self.logger.info("Índices creados en la base de datos")
        except Exception as e:
            self.logger.error(f"Error al conectar a la base de datos: {e}")
            raise

    async def save_order(self, exchange: str, order_id: str, symbol: str, order_type: str, status: str, timestamp: float):
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO orders (exchange, order_id, symbol, type, status, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (order_id) DO UPDATE
                    SET status = $5, timestamp = $6
                """, exchange, order_id, symbol, order_type, status, timestamp)
            self.logger.debug(f"Orden guardada: {order_id}")
        except Exception as e:
            self.logger.error(f"Error al guardar orden: {e}")

    async def save_report(self, date: str, total_profit: float, roi_percent: float, total_trades: int, report_data: dict, timestamp: float):
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO reports (date, total_profit, roi_percent, total_trades, report_data, timestamp)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (date) DO UPDATE
                    SET total_profit = $2, roi_percent = $3, total_trades = $4, report_data = $5, timestamp = $6
                """, date, total_profit, roi_percent, total_trades, report_data, timestamp)
            self.logger.debug(f"Informe guardado para la fecha: {date}")
        except Exception as e:
            self.logger.error(f"Error al guardar informe: {e}")

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
        self.logger.info("Conexión a la base de datos cerrada")
