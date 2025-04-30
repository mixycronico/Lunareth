import logging
import redis.asyncio as aioredis
from tenacity import retry, stop_after_attempt, wait_exponential


logger = logging.getLogger("corec.redis")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=10))
async def init_redis(redis_conf: dict) -> aioredis.Redis:
    try:
        url = f"redis://{redis_conf['username']}:{redis_conf['password']}@{redis_conf['host']}:{redis_conf['port']}"
        client = aioredis.from_url(url, decode_responses=True, max_connections=redis_conf.get("max_connections", 100))
        await client.ping()
        logger.info("🔴 Redis inicializado correctamente")
        return client
    except aioredis.RedisError as e:
        logger.error(f"[Redis] Error conectando: {e}")
        raise
