from sqlalchemy.orm import Session
from .audio_probe import AudioProcessingError, analyze_audio
from .file_registry import get_file
from .models import Job, Project
from .storage import absolute_from_relative


def process_analyze(session: Session, job: Job, progress):
    project = session.get(Project, job.project_id)
    if not project:
        raise AudioProcessingError("Project not found")
    file_item = get_file(session, job.request_json["source_file_id"], project.id)
    progress(10, "Reading media metadata")
    analysis = analyze_audio(absolute_from_relative(file_item.relative_path))
    progress(90, "Saving measured audio values")
    file_item.metadata_json = {**(file_item.metadata_json or {}), "analysis": analysis}
    project.analysis = analysis
    session.flush()
    return {"analysis": analysis, "file_id": file_item.id}
