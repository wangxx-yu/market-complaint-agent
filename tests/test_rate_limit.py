"""Rate Limit 中间件测试（Wave 7.2）。"""
from __future__ import annotations

import time

import pytest
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.core.rate_limit import RateLimitMiddleware


async def _ok(request):
    return PlainTextResponse("ok")


def _make_app(max_requests=5, whitelist=None):
    app = Starlette(
        routes=[Route("/api/test", _ok), Route("/health", _ok)],
        middleware=[Middleware(RateLimitMiddleware, max_requests=max_requests, whitelist_paths=whitelist or {"/health"})],
    )
    return app


class TestRateLimit:
    def test_within_limit(self):
        app = _make_app(max_requests=5)
        client = TestClient(app)
        for _ in range(3):
            resp = client.get("/api/test")
            assert resp.status_code == 200

    def test_exceeds_limit(self):
        app = _make_app(max_requests=3)
        client = TestClient(app)
        for _ in range(3):
            resp = client.get("/api/test")
            assert resp.status_code == 200
        resp = client.get("/api/test")
        assert resp.status_code == 429

    def test_whitelist_bypass(self):
        app = _make_app(max_requests=2, whitelist={"/health"})
        client = TestClient(app)
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200
