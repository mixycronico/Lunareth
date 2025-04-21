import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime, timedelta

class Scheduler:
    def __init__(self):
        self._sched = AsyncIOScheduler()
        self.logger = logging.getLogger("Scheduler")

    def start(self):
        """Inicia el scheduler; debe llamarse una sola vez al arrancar el plugin."""
        self._sched.start()
        self.logger.info("Scheduler iniciado")

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

    def schedule_cron(self, func, minute: int, hour: int, job_id: str, args=None, kwargs=None):
        """Programa una tarea diaria a una hora fija."""
        args = args or []
        kwargs = kwargs or {}
        trigger = CronTrigger(minute=minute, hour=hour)
        self._sched.add_job(
            func,
            trigger=trigger,
            id=job_id,
            replace_existing=True,
            misfire_grace_time=60,
            coalesce=True,
            args=args,
            kwargs=kwargs
        )
        self.logger.info(f"Tarea cron programada: {job_id}, a las {hour}:{minute}")
