from datetime import datetime, timezone
import logging
import signal
import time

from sqlalchemy import select

from .config import settings
from .database import Base, engine, session_scope
from .daw_export import process_daw_export
from .generation_jobs import process_generate
from .harmony_jobs import process_chords, process_midi
from .lyrics_jobs import process_lyrics
from .models import Job
from .process_analyze import process_analyze
from .render_jobs import process_master, process_mix
from .separation_jobs import process_separate
from .song_package_jobs import process_song_package

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("beatmaster-worker")
stop = False
PROCESSORS = {
    "analyze": process_analyze,
    "separate": process_separate,
    "mix": process_mix,
    "master": process_master,
    "generate": process_generate,
    "lyrics": process_lyrics,
    "chords": process_chords,
    "midi": process_midi,
    "daw_export": process_daw_export,
    "song_package": process_song_package,
}


def request_stop(*_):
    global stop
    stop = True


def now():
    return datetime.now(timezone.utc)


def claim_job():
    with session_scope() as session:
        job = session.scalar(select(Job).where(Job.status == "queued").order_by(Job.created_at.asc()).limit(1))
        if not job:
            return None
        job.status = "running"
        job.started_at = now()
        job.stage = "Starting"
        job.progress = 1
        session.flush()
        return job.id


def execute_job(job_id: str):
    with session_scope() as session:
        job = session.get(Job, job_id)
        if not job:
            return

        def progress(value: float, stage: str):
            job.progress = max(0, min(99, float(value)))
            job.stage = stage
            session.flush()
            session.commit()

        processor = PROCESSORS.get(job.type)
        if not processor:
            job.status = "failed"
            job.error = f"Unsupported job type: {job.type}"
            job.completed_at = now()
            return
        try:
            logger.info("Running %s job %s", job.type, job.id)
            job.result_json = processor(session, job, progress)
            job.status = "succeeded"
            job.progress = 100
            job.stage = "Completed"
            job.completed_at = now()
        except Exception as exc:
            logger.exception("Job %s failed", job.id)
            job.status = "failed"
            job.error = str(exc)[-12000:]
            job.stage = "Failed"
            job.completed_at = now()


def main():
    Base.metadata.create_all(bind=engine)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    logger.info("BeatMaster worker started")
    while not stop:
        job_id = claim_job()
        if job_id:
            execute_job(job_id)
        else:
            time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
