import psycopg2
import logging


logger = logging.getLogger("CoreCDB")


def init_postgresql(db_config: dict):
    """Inicializa la conexión a PostgreSQL."""
    try:
        conn = psycopg2.connect(**db_config)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bloques (
                id VARCHAR(50) PRIMARY KEY,
                canal INTEGER,
                num_entidades INTEGER,
                fitness FLOAT,
                timestamp FLOAT
            )
        """)
        conn.commit()
        cur.close()
        conn.close()
        logger.info("[DB] Tabla 'bloques' inicializada")
    except Exception as e:
        logger.error(f"[DB] Error inicializando PostgreSQL: {e}")
