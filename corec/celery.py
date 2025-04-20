from celery import Celery

# Configuración de la aplicación Celery
celery_app = Celery(
    'corec',
    broker='redis://localhost:6379/0',  # Ajusta según tu configuración de Redis
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
