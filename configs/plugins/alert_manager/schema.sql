-- configs/plugins/alert_manager/schema.sql
-- Esquema de base de datos para el plugin alert_manager

CREATE TABLE alerts (
    id SERIAL PRIMARY KEY,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    channel VARCHAR(50) NOT NULL,
    event_data JSONB NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_timestamp ON alerts(timestamp);