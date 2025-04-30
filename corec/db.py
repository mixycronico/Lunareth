import logging
import asyncpg


logger = logging.getLogger("CoreCDB")


async def init_postgresql(config: Dict[str, Any]):
    """Inicializa la conexión a PostgreSQL."""
    try:
        pool = await asyncpg.create_pool(**config)
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS bloques (
                    id VARCHAR(50) PRIMARY KEY,
                    canal INTEGER,
                    num_entidades INTEGER,
                    fitness FLOAT,
                    timestamp FLOAT
                );
                CREATE TABLE IF NOT EXISTS mensajes (
                    id SERIAL PRIMARY KEY,
                    bloque_id VARCHAR(50),
                    entidad_id VARCHAR(50),
                    canal INTEGER,
                    valor FLOAT,
                    clasificacion VARCHAR(50),
                    probabilidad FLOAT,
                    timestamp FLOAT
                );
            """)
            logger.info("[DB] Tablas 'bloques' y 'mensajes' inicializadas")
        return pool
    except Exception as e:
        logger.error(f"[DB] Error inicializando PostgreSQL: {e}")
        raise
