-- configs/plugins/system_analyzer/schema.sql
-- Esquema de base de datos para el plugin system_analyzer

CREATE TABLE insights (
    id SERIAL PRIMARY KEY,
    timestamp DOUBLE PRECISION NOT NULL,
    metrics JSONB NOT NULL,
    recommendations JSONB NOT NULL,
    analysis TEXT NOT NULL
);

CREATE INDEX idx_insights_timestamp ON insights(timestamp);