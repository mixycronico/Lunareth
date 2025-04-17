-- configs/plugins/user_management/schema.sql
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    email VARCHAR(100) NOT NULL UNIQUE,
    password VARCHAR(100) NOT NULL,
    name VARCHAR(100),
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    notification_preferences JSONB DEFAULT '{"email": false}',
    created_at DOUBLE PRECISION NOT NULL
);