"""The HTTP surface. It enqueues runs and reads run documents. It never runs the graph."""

import logging
from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from ra.config import get_settings
from ra.envelope import fail, ok
from ra.ids import new_run_id
from ra.ratelimit import BodySizeLimitMiddleware, RateLimitMiddleware
from ra.schemas import Budgets, RunState
from ra.store import RunStore, make_redis
from ra.worker import enqueue_run

log = logging.getLogger("ra.api")

QUESTION_MIN = 10
QUESTION_MAX = 1000


class ResearchRequest(BaseModel):
    question: str
    budgets: Budgets | None = None

    @field_validator("question")
    @classmethod
    def _sane_question(cls, v: str) -> str:
        v = v.strip()
        if not QUESTION_MIN <= len(v) <= QUESTION_MAX:
            raise ValueError(
                f"question must be {QUESTION_MIN} to {QUESTION_MAX} characters after trimming"
            )
        return v


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    redis = make_redis(settings.redis_url)
    pool = None
    try:
        # Inside the try from here on, so a pool that fails to open does not strand redis.
        pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
        app.state.settings = settings
        app.state.redis = redis
        app.state.store = RunStore(redis, lease_ttl_s=settings.lease_ttl_s)
        app.state.pool = pool
        yield
    finally:
        if pool is not None:
            await pool.aclose()
        await redis.aclose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="Research Agent", version="0.1.0", lifespan=lifespan)
    app.add_middleware(RateLimitMiddleware, per_minute=settings.rate_limit_per_min)
    # Outermost, so an oversized body is refused before anything tries to parse it.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_request_bytes)

    @app.exception_handler(HTTPException)
    async def _http_error(request: Request, exc: HTTPException):
        return JSONResponse(status_code=exc.status_code, content=fail(str(exc.detail)))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ())[1:]) or "body"
        return JSONResponse(
            status_code=422,
            content=fail(f"{field}: {first.get('msg', 'invalid request')}"),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content=fail("internal error"))

    @app.get("/healthz")
    async def healthz(request: Request):
        try:
            await request.app.state.redis.ping()
            redis_up = True
        except Exception as exc:
            log.warning("healthz could not reach redis: %s", exc)
            redis_up = False
        return JSONResponse(
            status_code=200 if redis_up else 503,
            content=ok({"ok": True, "redis": redis_up}) if redis_up else fail("redis unreachable"),
        )

    @app.post("/research", status_code=202)
    async def create_research(body: ResearchRequest, request: Request):
        store: RunStore = request.app.state.store
        state = RunState(
            run_id=new_run_id(),
            question=body.question,
            status="queued",
            budgets=body.budgets or Budgets(),
        )
        await store.save(state)
        await enqueue_run(request.app.state.pool, state)
        log.info("queued run %s", state.run_id)
        return ok({"run_id": state.run_id, "status": state.status})

    @app.get("/research/{run_id}")
    async def get_research(run_id: str, request: Request):
        state = await _load_or_404(request, run_id)
        return ok(
            {
                "run_id": state.run_id,
                "status": state.status,
                "question": state.question,
                "report_markdown": state.report_markdown,
                "cost_usd": state.cost_usd,
                "tokens_in": state.tokens_in,
                "tokens_out": state.tokens_out,
                "tavily_credits": state.tavily_credits,
                "created_at": state.created_at,
                "started_at": state.started_at,
                "finished_at": state.finished_at,
                "error": state.error,
            }
        )

    @app.get("/research/{run_id}/trace")
    async def get_trace(run_id: str, request: Request):
        state = await _load_or_404(request, run_id)
        return ok({"run_id": state.run_id, "steps": [s.model_dump() for s in state.steps]})

    return app


async def _load_or_404(request: Request, run_id: str) -> RunState:
    state = await request.app.state.store.load(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="no such run")
    return state


app = create_app()
