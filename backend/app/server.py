from contextlib import asynccontextmanager
from fastapi import FastAPI
from .api_files import router as files_router
from .api_processing import router as processing_router
from .api_projects import router as projects_router
from .api_system import router as system_router
from .database import Base, engine

@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

application = FastAPI(title="BeatMaster API", version="1.0.0", lifespan=lifespan)
application.include_router(system_router)
application.include_router(projects_router)
application.include_router(processing_router)
application.include_router(files_router)
