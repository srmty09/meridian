import os
import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import cost
import metrics
import router as router_mod
from cache import redis_cache, semantic_cache
from hf_client import generate

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Meridian", description="Learned LLM router with a verified semantic cache")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str


def _response_payload(*, response, model, cache_hit, cache_type, router_confidence, tokens):
    baseline = cost.baseline_cost(tokens)
    actual = 0.0 if cache_hit else cost.estimate_cost(model, tokens)
    return {
        "response": response,
        "model": model,
        "cache_hit": cache_hit,
        "cache_type": cache_type,
        "router_confidence": router_confidence,
        "estimated_cost": actual,
        "baseline_cost": baseline,
        "cost_saved": baseline - actual,
    }


@app.post("/chat")
def chat(req: ChatRequest):
    prompt = req.prompt.strip()
    started = time.perf_counter()

    exact = redis_cache.get(prompt)
    if exact:
        payload = _response_payload(
            response=exact["response"], model=exact["model"], cache_hit=True,
            cache_type="exact", router_confidence=None, tokens=exact["tokens"],
        )
        _record(payload, "exact", started)
        return payload

    sem = semantic_cache.lookup(prompt)
    if sem:
        payload = _response_payload(
            response=sem["response"], model=sem["model"], cache_hit=True,
            cache_type="semantic", router_confidence=None, tokens=sem["tokens"],
        )
        _record(payload, "semantic", started)
        return payload

    decision = router_mod.route(prompt)
    model = decision["model"]
    try:
        text, tokens = generate(prompt, model)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"inference failed via HF ({model}): {e}")

    redis_cache.put(prompt, text, model, tokens)
    semantic_cache.add(prompt, text, model, tokens)

    payload = _response_payload(
        response=text, model=model, cache_hit=False, cache_type=None,
        router_confidence=decision["confidence"], tokens=tokens,
    )
    payload["router"] = decision
    _record(payload, None, started)
    return payload


def _record(payload, cache_type, started):
    metrics.record_request(
        cache_type=cache_type,
        model=payload["model"],
        router_confidence=payload["router_confidence"],
        estimated_cost=payload["estimated_cost"],
        baseline_cost=payload["baseline_cost"],
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


@app.post("/route")
def route_only(req: ChatRequest):
    return router_mod.route(req.prompt.strip())


@app.post("/reset")
def reset_metrics():
    metrics.reset()
    return {"status": "reset"}


@app.get("/metrics")
def get_metrics():
    snap = metrics.snapshot()
    snap["semantic_cache_size"] = semantic_cache.size()
    snap["exact_cache"] = redis_cache.stats()
    return snap


@app.get("/health")
def health():
    return {
        "status": "ok",
        "router_model": router_mod.ROUTER_MODEL,
        "router_loaded": router_mod._model is not None,
        "exact_cache": redis_cache.stats(),
    }


@app.get("/adversarial-test")
def adversarial_test():
    summary = semantic_cache.run_adversarial_test()
    for _ in range(summary["false_hits"]):
        metrics.record_false_hit()
    return summary


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    @app.get("/")
    def index():
        return {"name": "Meridian", "docs": "/docs"}
