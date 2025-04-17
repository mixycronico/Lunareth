-- configs/plugins/daily_settlement/schema.sql
-- Esquema de base de datos para el plugin daily_settlement

CREATE TABLE reports (
    id SERIAL PRIMARY KEY,
    date VARCHAR(10) NOT NULL,
    total_profit FLOAT NOT NULL,
    roi_percent FLOAT NOT NULL,
    total_trades INTEGER NOT NULL,
    report_data JSONB NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE TABLE pool_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    total FLOAT NOT NULL DEFAULT 0,
    active_capital FLOAT NOT NULL DEFAULT 0,
    timestamp DOUBLE PRECISION NOT NULL,
    CONSTRAINT single_row CHECK (id = 1)
);

CREATE TABLE contributions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    amount FLOAT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE TABLE withdrawals (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) NOT NULL,
    amount FLOAT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_reports_date ON reports(date);
CREATE INDEX idx_reports_timestamp ON reports(timestamp);
CREATE INDEX idx_contributions_user_id ON contributions(user_id);
CREATE INDEX idx_withdrawals_user_id ON withdrawals(user_id);