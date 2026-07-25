"""
Simple in-memory per-IP sliding-window rate limiter.

Single-process, in-memory - fine for a single-instance deployment on one
box. If this ever runs as multiple worker processes/replicas, the limit
becomes per-process (each replica gets its own budget) rather than global;
a shared Redis-backed limiter would be needed at that point. Not built now
since this deployment is a single process on a single box.
"""
import time
from collections import defaultdict, deque

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60, exempt_paths: set[str] | None = None):
        super().__init__(app)
        self.limit = requests_per_minute
        self.window = 60.0
        self.exempt_paths = exempt_paths or {"/health", "/metrics"}
        self._hits: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        hits = self._hits[client_ip]
        while hits and now - hits[0] > self.window:
            hits.popleft()

        if len(hits) >= self.limit:
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit exceeded: {self.limit} requests/minute"},
                headers={"Retry-After": str(int(self.window - (now - hits[0])))},
            )

        hits.append(now)
        return await call_next(request)
