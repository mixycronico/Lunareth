-- configs/plugins/market_monitor/schema.sql
-- Esquema de base de datos para el plugin market_monitor

CREATE TABLE market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    price FLOAT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX idx_market_data_symbol ON market_data(symbol);