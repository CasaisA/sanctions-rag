"""HTTP surface. FastAPI is an optional dependency; the CLI works without it."""
from __future__ import annotations

import os
from typing import Any

import time

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from .agent import investigate, link
from .observability import METRICS, configure_logging, log, new_request_id, stage
from .pipeline import Pipeline
from .rag import answer as rag_answer

app = FastAPI(title="Sanctions research assistant", version="0.1.0",
              description="Hybrid retrieval, graph expansion and agentic investigation "
                          "over OpenSanctions data.")

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline.build(os.environ.get("SANCTIONS_RAG_DB", "data/sanctions.db"),
                                   os.environ.get("SANCTIONS_RAG_DSN"))
    return _pipeline


class SearchHit(BaseModel):
    id: str
    caption: str
    schema_: str = Field(alias="schema")
    countries: list[str] = []
    topics: list[str] = []
    datasets: list[str] = []
    score: float

    model_config = {"populate_by_name": True}


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=8, ge=1, le=50)
    expand: bool = True


class InvestigateRequest(AskRequest):
    max_iterations: int = Field(default=3, ge=1, le=6)


@app.middleware("http")
async def observe(request: Request, call_next):
    rid = new_request_id()
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = time.perf_counter() - t0
    METRICS.observe("http_request", dt)
    METRICS.count("http_request", f"{request.method} {request.url.path}")
    METRICS.count("http_status", str(response.status_code))
    log.info("request", extra={"extra_fields": {"method": request.method, "path": request.url.path,
                                                "status": response.status_code,
                                                "duration_ms": round(dt * 1000, 2)}})
    response.headers["x-request-id"] = rid
    return response


@app.get("/metrics", response_class=Response)
def metrics() -> Response:
    return Response(METRICS.render(), media_type="text/plain; version=0.0.4")


@app.on_event("startup")
def _warm() -> None:
    configure_logging()
    with stage("index_build"):
        get_pipeline()


@app.get("/health")
def health() -> dict[str, Any]:
    p = get_pipeline()
    return {"status": "ok", **p.store.counts()}


@app.get("/search", response_model=list[SearchHit])
def search(q: str = Query(min_length=2), k: int = Query(10, ge=1, le=50), expand: bool = False):
    p = get_pipeline()
    out = []
    with stage("retrieve", query_len=len(q), k=k, expand=expand):
        hits = p.search(q, k=k, expand=expand)
    for doc_id, score in hits:
        e = p.store.get(doc_id)
        if e is None:
            continue
        out.append(SearchHit(id=e.id, caption=e.caption, schema=e.schema, countries=e.countries,
                             topics=e.topics, datasets=e.datasets, score=round(float(score), 4)))
    return out


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    p = get_pipeline()
    hits = p.search(req.question, k=req.k, expand=req.expand)
    a = rag_answer(req.question, hits, p.store, limit=req.k)
    return {"question": a.question, "answer": a.text, "generator": a.generator,
            "evidence": [{"id": e.entity_id, "caption": e.caption, "datasets": e.datasets}
                         for e in a.evidence]}


@app.post("/investigate")
def investigate_endpoint(req: InvestigateRequest) -> dict[str, Any]:
    p = get_pipeline()
    return investigate(p, req.question, max_iterations=req.max_iterations,
                       k=req.k, expand=req.expand).as_dict()


@app.get("/link")
def link_endpoint(a: str = Query(min_length=2), b: str = Query(min_length=2), hops: int = 3):
    res = link(get_pipeline(), a, b, max_hops=hops)
    if not res.get("found") and res.get("reason", "").endswith("not found"):
        raise HTTPException(status_code=404, detail=res["reason"])
    return res


@app.get("/entity/{entity_id}")
def entity(entity_id: str) -> dict[str, Any]:
    p = get_pipeline()
    e = p.store.get(entity_id)
    if e is None:
        raise HTTPException(status_code=404, detail="unknown entity")
    nb = []
    for nid, schema, direction in p.store.neighbours(entity_id):
        n = p.store.get(nid)
        if n is not None:
            nb.append({"id": nid, "caption": n.caption, "relation": schema, "direction": direction})
    return {"id": e.id, "caption": e.caption, "schema": e.schema, "countries": e.countries,
            "topics": e.topics, "datasets": e.datasets, "properties": e.properties,
            "neighbours": nb}
