-- configs/plugins/cli_manager/schema.sql
-- Esquema de base de datos para el plugin cli_manager

CREATE TABLE actions (
    id SERIAL PRIMARY KEY,
    action VARCHAR(100) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE TABLE goals (
    goal_id VARCHAR(50) PRIMARY KEY,
    goal_data JSONB NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL
);

CREATE INDEX idx_actions_user_id ON actions(user_id);
CREATE INDEX idx_actions_timestamp ON actions(timestamp);
CREATE INDEX idx_goals_user_id ON goals(user_id);
CREATE INDEX idx_goals_timestamp ON goals(timestamp);