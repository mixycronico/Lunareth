-- configs/plugins/macro_sync/schema.sql
-- Esquema de base de datos para el plugin macro_sync

CREATE TABLE macro_metrics (
    id SERIAL PRIMARY KEY,
    sp500_price FLOAT,
    nasdaq_price FLOAT,
    vix_price FLOAT,
    gold_price FLOAT,
    oil_price FLOAT,
    altcoins_volume FLOAT,
    news_sentiment FLOAT,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_macro_metrics_timestamp ON macro_metrics(timestamp);