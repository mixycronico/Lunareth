import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta


class Scheduler:
    def __init__(self, nucleus=None):
        """Inicializa el scheduler para tareas periódicas.

        Args:
            nucleus: Instancia del núcleo de CoreC (opcional).
        """
        self._sched = AsyncIOScheduler()
        self.logger = logging.getLogger("CoreCScheduler")
        self.nucleus = nucleus

    def start(self):
        """Inicia el scheduler."""
        self._sched.start()
        self.logger.info("CoreC Scheduler iniciado")

    def shutdown(self):
        """Detiene el scheduler y todas sus tareas programadas."""
        try:
            self._sched.shutdown()
            self.logger.info("CoreC Scheduler detenido")
        except Exception as e:
            self.logger.error(f"Error deteniendo CoreC Scheduler: {e}")
            raise

    def schedule_periodic(self, func, seconds: int, job_id: str, start_delay: int = 0, args=None, kwargs=None):
        """Programa una tarea periódica cada N segundos.

        Args:
            func: Función a ejecutar periódicamente.
            seconds (int): Intervalo entre ejecuciones.
            job_id (str): Identificador único de la tarea.
            start_delay (int): Retraso inicial en segundos.
            args: Argumentos para la función.
            kwargs: Argumentos clave para la función.
        """
        args = args or []
        kwargs = kwargs or {}
        next_run = datetime.now() + timedelta(seconds=start_delay)
        self._sched.add_job(
            func,
            trigger=IntervalTrigger(seconds=seconds, start_date=next_run),
            id=job_id,
            replace_existing=True,
            misfire_grace_time=60,
            coalesce=True,
            args=args,
            kwargs=kwargs
        )
        self.logger.info(f"Tarea periódica programada: {job_id}, cada {seconds} segundos, inicio en {next_run}")
