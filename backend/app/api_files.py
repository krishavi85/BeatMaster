from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from .api_helpers import get_db
from .mime import audio_mime
from .models import AudioFile
from .storage import absolute_from_relative

router = APIRouter(prefix="/api/files", tags=["files"])

def response(file_id: str, session: Session, inline: bool):
    item = session.get(AudioFile, file_id)
    if not item:
        raise HTTPException(status_code=404, detail="Audio file not found")
    path = absolute_from_relative(item.relative_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio bytes not found")
    return FileResponse(path, media_type=item.mime_type or audio_mime(path), filename=item.original_name or path.name, content_disposition_type="inline" if inline else "attachment")

@router.get("/{file_id}/stream")
def stream_file(file_id: str, session: Session = Depends(get_db)):
    return response(file_id, session, True)

@router.get("/{file_id}/download")
def download_file(file_id: str, session: Session = Depends(get_db)):
    return response(file_id, session, False)
