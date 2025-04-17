-- configs/plugins/capital_pool/schema.sql
-- Esquema de base de datos para el plugin capital_pool

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

CREATE TABLE pool_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    total FLOAT NOT NULL DEFAULT 0,
    active_capital FLOAT NOT NULL DEFAULT 0,
    timestamp DOUBLE PRECISION NOT NULL,
    CONSTRAINT single_row CHECK (id = 1)
);

CREATE INDEX idx_contributions_user_id ON contributions(user_id);
CREATE INDEX idx_contributions_timestamp ON contributions(timestamp);
CREATE INDEX idx_withdrawals_user_id ON withdrawals(user_id);
CREATE INDEX idx_withdrawals_timestamp ON withdrawals(timestamp);