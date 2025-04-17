CREATE TABLE nodos (
  nodo_id VARCHAR(50),
  instance_id VARCHAR(50),
  ultima_actividad DOUBLE PRECISION,
  carga DOUBLE PRECISION,
  es_espejo BOOLEAN,
  original_id VARCHAR(50),
  PRIMARY KEY (nodo_id, instance_id)
);

CREATE TABLE eventos (
  id SERIAL PRIMARY KEY,
  canal VARCHAR(50),
  datos BYTEA,
  timestamp DOUBLE PRECISION,
  instance_id VARCHAR(50)
);

CREATE TABLE auditoria (
  evento_id VARCHAR(50) PRIMARY KEY,
  tipo VARCHAR(50),
  detalle TEXT,
  timestamp DOUBLE PRECISION,
  instance_id VARCHAR(50)
);

CREATE INDEX idx_eventos_canal ON eventos(canal);
CREATE INDEX idx_eventos_timestamp ON eventos(timestamp);