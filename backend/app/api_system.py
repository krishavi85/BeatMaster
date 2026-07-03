import importlib.util
import os
import shutil
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .api_helpers import get_db
from .config import settings
from .dashboard import render_dashboard
from .pages import page
from .schemas import CapabilityOut

router = APIRouter(tags=["system"])

@router.get("/", include_in_schema=False)
def dashboard(session: Session = Depends(get_db)):
    return page("Dashboard", render_dashboard(session))

@router.get("/health")
def health():
    return {"status": "ok", "service": "BeatMaster API", "version": "1.0.0"}

@router.get("/api/capabilities", response_model=CapabilityOut)
def capabilities():
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    return CapabilityOut(ffmpeg=shutil.which(settings.ffmpeg_bin) is not None, ffprobe=shutil.which(settings.ffprobe_bin) is not None, demucs=importlib.util.find_spec("demucs") is not None, musicgen_enabled=settings.enable_local_musicgen, musicgen_runtime=importlib.util.find_spec("transformers") is not None and importlib.util.find_spec("torch") is not None, cuda_available=cuda, separation_models=["htdemucs", "htdemucs_ft", "htdemucs_6s", "mdx_extra_q"], generation_model=os.getenv("MUSICGEN_MODEL", "not configured"))
