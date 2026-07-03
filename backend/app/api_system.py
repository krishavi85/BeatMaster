from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .api_helpers import get_db
from .culture_profiles import SUPPORTED_LANGUAGES, list_profiles
from .dashboard_full import render_dashboard
from .model_registry import registry_status
from .pages import page
from .runtime_capabilities import inspect_capabilities
from .schemas import CapabilityOut
from .ui_jobs import router as jobs_ui
from .ui_processing import router as processing_ui
from .ui_projects import router as projects_ui

router = APIRouter(tags=["system"])
router.include_router(projects_ui)
router.include_router(processing_ui)
router.include_router(jobs_ui)


@router.get("/", include_in_schema=False)
def dashboard(session: Session = Depends(get_db)):
    return page("Dashboard", render_dashboard(session))


@router.get("/health")
def health():
    return {"status": "ok", "service": "BeatMaster API", "version": "1.1.0"}


@router.get("/api/culture-profiles")
def culture_profiles():
    return {"profiles": list_profiles(), "languages": SUPPORTED_LANGUAGES, "model_registry": registry_status()}


@router.get("/api/languages")
def languages():
    return {"languages": SUPPORTED_LANGUAGES}


@router.get("/api/capabilities", response_model=CapabilityOut)
def capabilities():
    return CapabilityOut(**inspect_capabilities())
