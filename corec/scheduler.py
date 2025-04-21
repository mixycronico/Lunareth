import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta

class Scheduler:
    def __init__(self):
        self._sched = AsyncIOScheduler()
        self.logger = logging.getLogger("CoreCScheduler")

    def start(self):
        """Inicia el scheduler; debe llamarse una sola vez al arrancar CoreC."""
        self._sched.start()
        self.logger.info("CoreC Scheduler iniciado")

    def schedule_periodic(self, func, seconds: int, job_id: str, start_delay: int = 0, args=None, kwargs=None):
        """Programa una tarea periódica cada N segundos."""
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
