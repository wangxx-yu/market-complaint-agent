# ── 市场监管投诉智能处理系统 ──
# 本地优先、人机协同的多智能体 MVP
#
# 构建: docker build -t market-complaint-agent .
# 运行: docker compose up

FROM python:3.12-slim

WORKDIR /app

# 系统依赖（chromadb 需要 sqlite3, sentence-transformers 需要 gcc 等）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖（利用 Docker 缓存层）
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]" && \
    pip install --no-cache-dir structlog prometheus-client

# 复制项目源码
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/
COPY tests/ ./tests/

# 暴露 FastAPI 端口
EXPOSE 8000

# 默认启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
