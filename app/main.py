from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .core import STATIC_DIR
from .routers import api as api_router
from .routers import pages as pages_router

APP_TITLE = "Finance Stock Prediction API"

app = FastAPI(title=APP_TITLE)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Include routers
app.include_router(api_router.router)
app.include_router(pages_router.router)
