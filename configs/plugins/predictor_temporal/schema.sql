-- configs/plugins/predictor_temporal/schema.sql
-- Esquema de base de datos para el plugin predictor_temporal

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    nano_id VARCHAR(50) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    prediction JSONB NOT NULL,
    actual_value FLOAT,
    error FLOAT,
    macro_context JSONB,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_predictions_symbol ON predictions(symbol);

CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    nano_id VARCHAR(50) NOT NULL,
    mse FLOAT NOT NULL,
    mae FLOAT NOT NULL,
    predictions_count INTEGER NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);