from .generation_jobs import generate_audio as generate_fallback
from .own_model import generate as generate_own


def generate_audio(request, project, job_id, progress):
    result = generate_own(request, project, job_id, progress)
    if result is not None:
        return result
    return generate_fallback(request, project, job_id, progress)
