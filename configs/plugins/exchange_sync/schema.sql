-- configs/plugins/exchange_sync/schema.sql
-- Esquema de base de datos para el plugin exchange_sync

CREATE TABLE exchange_data (
    id SERIAL PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    market VARCHAR(20) NOT NULL,
    price FLOAT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE TABLE open_orders (
    id SERIAL PRIMARY KEY,
    exchange VARCHAR(50) NOT NULL,
    order_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    market VARCHAR(20) NOT NULL,
    status VARCHAR(50) NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_exchange_data_timestamp ON exchange_data(timestamp);
CREATE INDEX idx_exchange_data_symbol ON exchange_data(symbol);
CREATE INDEX idx_open_orders_timestamp ON open_orders(timestamp);
CREATE INDEX idx_open_orders_order_id ON open_orders(order_id);