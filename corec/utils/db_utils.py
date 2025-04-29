import logging
import asyncpg
import redis.asyncio as aioredis
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger("CoreCDB")


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(asyncpg.PostgresError),
    before_sleep=lambda retry_state: logger.info(
        f"Reintentando conexión a PostgreSQL... Intento {retry_state.attempt_number}"
    )
)
async def init_postgresql(config: dict) -> asyncpg.Pool:
    """Inicializa el pool de conexiones a PostgreSQL y crea tablas necesarias.

    Args:
        config (dict): Configuración de la base de datos (dbname, user, password, host, port).

    Returns:
        asyncpg.Pool: Pool de conexiones inicializado.

    Raises:
        asyncpg.PostgresError: Si falla la conexión a PostgreSQL.
    """
    config = config.copy()
    if "dbname" in config:
        config["database"] = config.pop("dbname")
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
                timestamp FLOAT,
                roles JSONB
            );
            CREATE TABLE IF NOT EXISTS alertas (
                id SERIAL PRIMARY KEY,
                tipo VARCHAR(100),
                bloque_id VARCHAR(50),
                mensaje TEXT,
                timestamp FLOAT,
                datos JSONB
            );
            CREATE TABLE IF NOT EXISTS aprendizajes (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                bloque_id VARCHAR(50),
                estrategia JSONB,
                fitness FLOAT,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS enlaces (
                id SERIAL PRIMARY KEY,
                entidad_a VARCHAR(50),
                entidad_b VARCHAR(50),
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS entidades (
                id SERIAL PRIMARY KEY,
                entidad_id VARCHAR(50),
                bloque_id VARCHAR(50),
                roles JSONB,
                quantization_step FLOAT,
                min_fitness FLOAT,
                mutation_rate FLOAT,
                timestamp FLOAT
            );
        """)
        logger.info("Tablas 'bloques', 'mensajes', 'alertas', 'aprendizajes', 'enlaces', 'entidades' inicializadas")
    return pool


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(aioredis.RedisError),
    before_sleep=lambda retry_state: logger.info(
        f"Reintentando conexión a Redis... Intento {retry_state.attempt_number}"
    )
)
async def init_redis(redis_conf: dict) -> aioredis.Redis:
    """Inicializa y devuelve un cliente Redis asíncrono.

    Args:
        redis_conf (dict): Configuración de Redis (host, port, username, password, max_connections).

    Returns:
        aioredis.Redis: Cliente Redis inicializado.

    Raises:
        aioredis.RedisError: Si falla la conexión a Redis.
    """
    url = f"redis://{redis_conf['username']}:{redis_conf['password']}@{redis_conf['host']}:{redis_conf['port']}"
    client = aioredis.from_url(url, decode_responses=True, max_connections=redis_conf.get("max_connections", 100))
    await client.ping()
    logger.info("Redis inicializado correctamente")
    return client
