-- configs/plugins/trading_execution/schema.sql
-- Esquema de base de datos para el plugin trading_execution

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    order_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    market VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    quantity FLOAT NOT NULL,
    price FLOAT NOT NULL,
    status VARCHAR(20) NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL,
    close_reason VARCHAR(50),
    close_timestamp DOUBLE PRECISION
);

CREATE INDEX idx_orders_timestamp ON orders(timestamp);
CREATE INDEX idx_orders_order_id ON orders(order_id);