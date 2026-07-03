from sqlalchemy.orm import Session
from .models import Job


def process_lyrics(session: Session, job: Job, progress):
    raise RuntimeError("Lyrics provider is not configured")
