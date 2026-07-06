from .audio_probe import analyze_audio
from .file_registry import register_file
from .generation_entry import generate_audio
from .models import Project


def process_generate(session, job, progress):
    request = job.request_json
    project = session.get(Project, job.project_id) if job.project_id else None
    if project is None:
        project = Project(name=request.get("name") or "Generated music")
        session.add(project)
        session.flush()
        job.project_id = project.id
    output, metadata = generate_audio(request, project, job.id, progress)
    analysis = analyze_audio(output)
    item = register_file(session, project.id, output, "generated", request.get("name") or "Generated music")
    return {"file_id": item.id, "project_id": project.id, "analysis": analysis, "metadata": metadata}
