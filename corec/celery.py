from celery import Celery

# Configuración de la aplicación Celery
celery_app = Celery(
    'corec',
    broker='redis://corec_user:secure_password@localhost:6379/0',
    backend='redis://corec_user:secure_password@localhost:6379/0',
    include=['corec.modules.ejecucion']
)

# Configuración adicional de Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)


def configure_celery(redis_config: dict):
    """Configura Celery con las credenciales de Redis proporcionadas.

    Args:
        redis_config (dict): Configuración de Redis (host, port, username, password).
    """
    broker_url = f"redis://{redis_config['username']}:{redis_config['password']}@{redis_config['host']}:{redis_config['port']}/0"
    celery_app.conf.update(
        broker=broker_url,
        result_backend=broker_url
    )
