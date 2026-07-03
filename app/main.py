from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.routes import router

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="市场监管投诉智能处理系统",
    version="0.1.0",
    description="本地优先、人机协同的多智能体投诉处理系统 MVP。",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
app.include_router(router)


@app.get("/")
def review_workbench() -> FileResponse:
    return FileResponse(BASE_DIR / "app" / "static" / "review.html")
