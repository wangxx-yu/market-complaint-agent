from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="市场监管投诉智能处理系统",
    version="0.1.0",
    description="本地优先、人机协同的多智能体投诉处理系统 MVP。",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
def readyz() -> dict:
    checks: dict[str, str] = {}
    # ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        _ = client.list_collections()
        checks["chromadb"] = "ok"
    except Exception as e:
        checks["chromadb"] = f"error: {e}"
    # 模型文件
    model_path = Path("models/accept_v4/accept_model.joblib")
    checks["accept_model"] = "ok" if model_path.exists() else "missing"
    # 数据目录
    data_dir = Path("data")
    checks["data_dir"] = "ok" if data_dir.exists() else "missing"
    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ready" if all_ok else "degraded", "checks": checks}


# Rate Limit 中间件
from app.core.rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware, whitelist_paths={"/health", "/readyz", "/metrics"})


app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
app.include_router(router)


@app.get("/")
def review_workbench() -> FileResponse:
    return FileResponse(BASE_DIR / "app" / "static" / "review.html")
