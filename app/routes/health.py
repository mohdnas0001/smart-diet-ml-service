import time
from fastapi import APIRouter, Request
from app.config import settings

router = APIRouter()
_start_time = time.time()


@router.get("/api/health")
async def health(request: Request):
    uptime = round(time.time() - _start_time, 2)
    pipeline = getattr(request.app.state, "pipeline", None)
    demo = getattr(pipeline, "demo_mode", True) if pipeline else True
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "uptime_seconds": uptime,
        "model_status": "demo" if demo else "loaded",
        "demo_mode": demo,
    }
