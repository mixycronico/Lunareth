# corec/db.py
import psycopg2
import time
import logging

logger = logging.getLogger("corec.db")

def init_postgresql(db_config: dict):
    """
    Crea la tabla particionada 'bloques' e índices si no existen.
    Llama a este método al arrancar la app.
    """
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()
    # Tabla principal particionada por timestamp
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bloques (
            id TEXT PRIMARY KEY,
            canal TEXT,
            num_entidades INTEGER,
            fitness REAL,
            timestamp DOUBLE PRECISION,
            instance_id TEXT
        ) PARTITION BY RANGE (timestamp)
    """)
    # Ejemplo de partición mensual
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bloques_2025_04 PARTITION OF bloques
        FOR VALUES FROM (%s) TO (%s)
    """, (
        time.mktime((2025,4,1,0,0,0,0,0,0)),
        time.mktime((2025,5,1,0,0,0,0,0,0))
    ))
    cur.execute("CREATE INDEX IF NOT EXISTS idx_bloques_canal_timestamp ON bloques (canal, timestamp DESC)")
    conn.commit()
    cur.close()
    conn.close()
    logger.info("🗄️ PostgreSQL inicializado correctamente")