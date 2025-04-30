from celery import Celery
from corec.config_loader import load_config

def configure_celery(config_path: str = "config/corec_config.json"):
    """Configura Celery con las credenciales de Redis proporcionadas."""
    config = load_config(config_path).redis_config.model_dump()
    broker_url = f"redis://{config['username']}:{config['password']}@{config['host']}:{config['port']}/0"
    celery_app.conf.update(
        broker=broker_url,
        result_backend=broker_url
    )

# Configuración de la aplicación Celery
celery_app = Celery(
    'corec',
    broker='redis://localhost:6379/0',  # Valor por defecto, se sobrescribe
    backend='redis://localhost:6379/0',
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

# Cargar configuración desde el archivo
configure_celery()
