# corec/redis_client.py
import logging
import redis.asyncio as aioredis

logger = logging.getLogger("corec.redis")

async def init_redis(redis_conf: dict):
    """
    Inicializa y devuelve un cliente Redis asíncrono.
    redis_conf debe tener host, port, username, password.
    """
    url = f"redis://{redis_conf['username']}:{redis_conf['password']}@{redis_conf['host']}:{redis_conf['port']}"
    client = await aioredis.from_url(url, decode_responses=True)
    logger.info("🔴 Redis inicializado correctamente")
    return client