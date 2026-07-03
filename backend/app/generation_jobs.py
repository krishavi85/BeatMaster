from sqlalchemy.orm import Session
from .audio_probe import AudioProcessingError
from .models import Job


def process_generate(session: Session, job: Job, progress):
    raise AudioProcessingError("Configure a licensed generation provider before enabling this processor")
