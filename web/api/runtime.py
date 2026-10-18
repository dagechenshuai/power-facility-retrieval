"""Web 服务通用的安全与运行时配置。"""

import os
import secrets
import uuid

from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


def configure_runtime(app) -> None:
    """配置跨域、请求追踪和可选 API Key 鉴权。"""
    origins = [
        value.strip()
        for value in os.getenv(
            "CORS_ORIGINS", "http://127.0.0.1:8002,http://127.0.0.1:8003"
        ).split(",")
        if value.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Last-Event-ID", "X-API-Key"],
    )

    @app.middleware("http")
    async def secure_and_trace(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        expected_key = os.getenv("AGENT_API_KEY", "").strip()
        public_paths = {"/", "/chat.html", "/import.html", "/health", "/ready"}
        if expected_key and request.url.path not in public_paths:
            authorization = request.headers.get("Authorization", "")
            supplied_key = (
                authorization.removeprefix("Bearer ").strip()
                or request.headers.get("X-API-Key", "").strip()
                or request.query_params.get("api_key", "").strip()
            )
            if not secrets.compare_digest(supplied_key, expected_key):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "未授权", "request_id": request_id},
                    headers={"X-Request-ID": request_id},
                )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
