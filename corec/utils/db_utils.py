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
    """Inicializa el pool de conexiones a PostgreSQL y crea tablas necesarias."""
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
            CREATE TABLE IF NOT EXISTS cognitivo_memoria (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                tipo VARCHAR(100),
                memoria JSONB,
                intuiciones JSONB,
                percepciones JSONB,
                decisiones JSONB,
                decisiones_fallidas JSONB,
                contexto JSONB,
                memoria_semantica JSONB,
                yo JSONB,
                intenciones JSONB,
                atencion JSONB,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS cognitivo_decisiones_fallidas (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                opcion VARCHAR(100),
                confianza FLOAT,
                motivo_fallo TEXT,
                contexto JSONB,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS cognitivo_memoria_semantica (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                concepto_a VARCHAR(100),
                concepto_b VARCHAR(100),
                relacion VARCHAR(50),
                peso FLOAT,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS cognitivo_autorreferencias (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                estado JSONB,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS cognitivo_yo (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                estado JSONB,
                memoria JSONB,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS cognitivo_cambios_yo (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                cambio JSONB,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS cognitivo_metadialogo (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                afirmacion TEXT,
                contexto JSONB,
                referencias JSONB,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS cognitivo_intenciones (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                intencion JSONB,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS cognitivo_contradicciones (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                contradiccion JSONB,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS cognitivo_atencion (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                atencion JSONB,
                timestamp FLOAT
            );
            CREATE TABLE IF NOT EXISTS cognitivo_conflictos (
                id SERIAL PRIMARY KEY,
                instancia_id VARCHAR(50),
                conflicto JSONB,
                timestamp FLOAT
            );
        """)
        logger.info("Tablas inicializadas")
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
    """Inicializa y devuelve un cliente Redis asíncrono."""
    url = f"redis://{redis_conf['username']}:{redis_conf['password']}@{redis_conf['host']}:{redis_conf['port']}"
    client = aioredis.from_url(url, decode_responses=True, max_connections=redis_conf.get("max_connections", 100))
    await client.ping()
    logger.info("Redis inicializado correctamente")
    return client
