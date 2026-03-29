"""
Pi Memory — Embedding & Retrieval Service

Serves embeddings from GTE-Qwen2-7B-instruct and manages the Qdrant
vector index for Pi's associative memory system.
"""

import os
import math
import time
import uuid
import logging
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pi-memory")

# --- Globals (populated on startup) ---
model = None
tokenizer = None
qdrant = None

COLLECTION = "memories"
MODEL_DIM = 3584          # GTE-Qwen2-7B base dimensions
STATE_DIM = 6             # certainty, clarity, scope, stakes, valence, arousal
TOTAL_DIM = MODEL_DIM + STATE_DIM  # 3590


# --- Activity Log ---

MAX_ACTIVITY_LOG = 100

activity_log = deque(maxlen=MAX_ACTIVITY_LOG)


def log_activity(operation: str, details: dict, elapsed_ms: float):
    """Record an operation for the /activity endpoint."""
    entry = {
        "timestamp": time.time(),
        "operation": operation,
        "elapsed_ms": round(elapsed_ms, 2),
        **details,
    }
    activity_log.appendleft(entry)
    logger.info(f"[{operation}] {elapsed_ms:.1f}ms — {details}")


# --- Helpers ---

def make_point_id(memory_id: str, ctx_index: int) -> str:
    """Deterministic UUID from memory_id:ctx_index for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{memory_id}:{ctx_index}"))


def compute_strength(initial_strength: float, access_count: int,
                     decay_rate: float, last_accessed: float) -> float:
    """Temporal strength: decays over time, boosted by access.
    decay_rate is per-day (0.05 = ~5% per day). A memory accessed
    yesterday is at ~95% strength. Unretrieved for a month: ~22%."""
    age_days = (time.time() - last_accessed) / 86400.0
    return (
        initial_strength
        * (1 + math.log(max(access_count, 1)))
        * math.exp(-decay_rate * age_days)
    )


# --- Request/Response Models ---

class EmbedRequest(BaseModel):
    content: str
    context: str = ""
    active: str = ""
    purposes: str = ""
    certainty: float = Field(0.5, ge=0.0, le=1.0)
    clarity: float = Field(0.5, ge=0.0, le=1.0)
    scope: float = Field(0.5, ge=0.0, le=1.0)
    stakes: float = Field(0.5, ge=0.0, le=1.0)
    valence: float = Field(0.5, ge=0.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)


class EmbedResponse(BaseModel):
    vector: list[float]
    elapsed_ms: float


class BatchEmbedRequest(BaseModel):
    inputs: list[EmbedRequest]


class BatchEmbedResponse(BaseModel):
    vectors: list[list[float]]
    elapsed_ms: float


class StoreRequest(BaseModel):
    memory_id: str
    ctx_index: int = 0
    content: str
    context_summary: str = ""
    embed: EmbedRequest
    initial_strength: float = 1.0
    decay_rate: float = 0.05
    created_at: float = Field(default_factory=time.time)
    last_accessed: float = Field(default_factory=time.time)
    access_count: int = 0
    created_at: float
    co_occurrence: dict[str, int] = Field(default_factory=dict)


class StoreResponse(BaseModel):
    point_id: str
    elapsed_ms: float


class SearchRequest(BaseModel):
    query: EmbedRequest
    top_k: int = 10
    exclude_memory_ids: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    memory_id: str
    ctx_index: int
    content: str
    context_summary: str
    score: float           # weighted score (raw_similarity * strength)
    raw_similarity: float  # raw cosine similarity from Qdrant
    strength: float        # temporal strength at retrieval time
    initial_strength: float
    decay_rate: float
    last_accessed: float
    access_count: int
    created_at: float
    co_occurrence: dict[str, int]


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query_elapsed_ms: float    # time to embed the query
    search_elapsed_ms: float   # time for Qdrant search
    total_elapsed_ms: float    # total


class UpdateDynamicsRequest(BaseModel):
    retrieved: list[dict]  # [{memory_id, ctx_index}]
    lr: float = 0.01


class UpdateDynamicsResponse(BaseModel):
    updated: int
    elapsed_ms: float


# --- Model Loading ---

def load_model():
    global model, tokenizer

    model_name = os.environ.get("MODEL_NAME", "Alibaba-NLP/gte-Qwen2-7B-instruct")
    device = os.environ.get("DEVICE", "cuda:0")
    quantize = os.environ.get("QUANTIZE", "int8")
    cache_dir = "/models"

    logger.info(f"Loading {model_name} ({quantize}) on {device}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )

    if quantize == "int8":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=device,
        )
    else:
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device,
        )

    model.eval()
    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")


def init_qdrant():
    global qdrant

    qdrant_host = os.environ.get("QDRANT_HOST", "qdrant")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))

    logger.info(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION not in collections:
        logger.info(f"Creating collection '{COLLECTION}' ({TOTAL_DIM}d)...")
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(
                size=TOTAL_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Collection created.")
    else:
        logger.info(f"Collection '{COLLECTION}' exists.")


# --- Embedding Logic ---

def format_structured_input(req: EmbedRequest, is_query: bool = False) -> str:
    if is_query:
        # GTE-Qwen2 requires Instruct prefix for queries
        parts = [f"Instruct: Retrieve relevant memories for the current context."]
        parts.append(f"Query: {req.content}")
    else:
        parts = [f"{req.content}"]
    if req.context:
        parts.append(f"[CONTEXT: {req.context}]")
    if req.active:
        parts.append(f"[ACTIVE: {req.active}]")
    if req.purposes:
        parts.append(f"[PURPOSES: {req.purposes}]")
    parts.append(f"[CERTAINTY: {req.certainty:.2f}]")
    parts.append(f"[CLARITY: {req.clarity:.2f}]")
    parts.append(f"[SCOPE: {req.scope:.2f}]")
    parts.append(f"[STAKES: {req.stakes:.2f}]")
    parts.append(f"[VALENCE: {req.valence:.2f}]")
    parts.append(f"[AROUSAL: {req.arousal:.2f}]")
    return "\n".join(parts)


def compute_embedding(text: str, state_dims: list[float]) -> list[float]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Last token pooling (required by GTE-Qwen2)
    attention_mask = inputs["attention_mask"]
    hidden = outputs.last_hidden_state
    # Find the last non-padding token for each sequence
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = hidden.shape[0]
    pooled = hidden[torch.arange(batch_size, device=hidden.device), sequence_lengths]

    # Normalize
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1).squeeze(0)

    # Convert to list and append state dimensions
    # State dims are centered at 0 and scaled to ~10% of content magnitude.
    # When all states are 0.5 (default), they become 0.0 — no effect on similarity.
    # When they differ, they contribute a meaningful but not dominant signal.
    STATE_SCALE = 0.1
    base_vec = pooled.cpu().float().numpy().tolist()
    scaled_state = [(v - 0.5) * STATE_SCALE for v in state_dims]
    return base_vec + scaled_state


def embed_request(req: EmbedRequest, is_query: bool = False) -> list[float]:
    text = format_structured_input(req, is_query=is_query)
    state_dims = [req.certainty, req.clarity, req.scope, req.stakes,
                  req.valence, req.arousal]
    return compute_embedding(text, state_dims)


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    init_qdrant()
    logger.info("Pi Memory service ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Pi Memory Service",
    description="Embedding and retrieval for Pi's associative memory system",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    # Count memories
    try:
        info = qdrant.get_collection(COLLECTION)
        count = info.points_count
    except Exception:
        count = -1

    return {
        "status": "ok",
        "model": os.environ.get("MODEL_NAME", "unknown"),
        "collection": COLLECTION,
        "total_dim": TOTAL_DIM,
        "memory_count": count,
    }


@app.get("/activity")
async def get_activity(limit: int = 20):
    """Recent operations log for observability."""
    return {"operations": list(activity_log)[:limit]}


class EmbedEndpointRequest(BaseModel):
    """Wraps EmbedRequest with optional is_query flag."""
    content: str
    context: str = ""
    active: str = ""
    purposes: str = ""
    certainty: float = Field(0.5, ge=0.0, le=1.0)
    clarity: float = Field(0.5, ge=0.0, le=1.0)
    scope: float = Field(0.5, ge=0.0, le=1.0)
    stakes: float = Field(0.5, ge=0.0, le=1.0)
    valence: float = Field(0.5, ge=0.0, le=1.0)
    arousal: float = Field(0.5, ge=0.0, le=1.0)
    is_query: bool = False


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedEndpointRequest):
    start = time.time()
    embed_req = EmbedRequest(**{k: v for k, v in req.model_dump().items() if k != "is_query"})
    vector = embed_request(embed_req, is_query=req.is_query)
    elapsed_ms = (time.time() - start) * 1000

    log_activity("embed", {
        "content_preview": req.content[:80],
        "state": [req.certainty, req.clarity, req.scope, req.stakes, req.valence, req.arousal],
    }, elapsed_ms)

    return EmbedResponse(vector=vector, elapsed_ms=elapsed_ms)


@app.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_batch(req: BatchEmbedRequest):
    start = time.time()
    vectors = [embed_request(r) for r in req.inputs]
    elapsed_ms = (time.time() - start) * 1000

    log_activity("embed_batch", {"count": len(req.inputs)}, elapsed_ms)

    return BatchEmbedResponse(vectors=vectors, elapsed_ms=elapsed_ms)


@app.post("/store", response_model=StoreResponse)
async def store(req: StoreRequest):
    start = time.time()

    vector = embed_request(req.embed)
    point_id = make_point_id(req.memory_id, req.ctx_index)

    offset = [0.0] * TOTAL_DIM

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "memory_id": req.memory_id,
                    "ctx_index": req.ctx_index,
                    "content": req.content,
                    "context_summary": req.context_summary,
                    "initial_strength": req.initial_strength,
                    "decay_rate": req.decay_rate,
                    "created_at": req.created_at,
                    "last_accessed": req.last_accessed,
                    "access_count": req.access_count,
                    "co_occurrence": req.co_occurrence,
                    "base_embedding": vector,
                    "offset": offset,
                },
            )
        ],
    )

    elapsed_ms = (time.time() - start) * 1000

    log_activity("store", {
        "memory_id": req.memory_id,
        "content_preview": req.content[:80],
        "initial_strength": req.initial_strength,
    }, elapsed_ms)

    return StoreResponse(point_id=point_id, elapsed_ms=elapsed_ms)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    start = time.time()

    # Step 1: Embed query
    t_embed = time.time()
    query_vector = embed_request(req.query, is_query=True)
    query_elapsed_ms = (time.time() - t_embed) * 1000

    # Step 2: Search Qdrant
    t_search = time.time()
    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=req.top_k * 3 if req.exclude_memory_ids else req.top_k,
        with_payload=True,
    )
    search_elapsed_ms = (time.time() - t_search) * 1000

    # Step 3: Score and deduplicate
    search_results = []
    seen_memory_ids = set()

    for hit in response.points:
        payload = hit.payload
        mem_id = payload["memory_id"]

        if mem_id in req.exclude_memory_ids:
            continue
        if mem_id in seen_memory_ids:
            continue
        seen_memory_ids.add(mem_id)

        raw_similarity = hit.score
        strength = compute_strength(
            payload.get("initial_strength", 1.0),
            payload.get("access_count", 0),
            payload.get("decay_rate", 0.05),
            payload.get("last_accessed", time.time()),
        )
        weighted_score = raw_similarity * strength

        search_results.append(SearchResult(
            memory_id=mem_id,
            ctx_index=payload.get("ctx_index", 0),
            content=payload.get("content", ""),
            context_summary=payload.get("context_summary", ""),
            score=weighted_score,
            raw_similarity=raw_similarity,
            strength=strength,
            initial_strength=payload.get("initial_strength", 1.0),
            decay_rate=payload.get("decay_rate", 0.05),
            last_accessed=payload.get("last_accessed", 0),
            access_count=payload.get("access_count", 0),
            created_at=payload.get("created_at", 0),
            co_occurrence=payload.get("co_occurrence", {}),
        ))

    search_results.sort(key=lambda r: r.score, reverse=True)
    search_results = search_results[:req.top_k]

    total_elapsed_ms = (time.time() - start) * 1000

    # Log with detailed scoring breakdown
    result_details = [
        {
            "memory_id": r.memory_id,
            "content_preview": r.content[:50],
            "raw_similarity": round(r.raw_similarity, 4),
            "strength": round(r.strength, 4),
            "weighted_score": round(r.score, 4),
        }
        for r in search_results
    ]

    log_activity("search", {
        "query_preview": req.query.content[:80],
        "state": [req.query.certainty, req.query.clarity, req.query.scope,
                  req.query.stakes, req.query.valence, req.query.arousal],
        "candidates": len(response.points),
        "returned": len(search_results),
        "results": result_details,
        "query_ms": round(query_elapsed_ms, 1),
        "search_ms": round(search_elapsed_ms, 1),
    }, total_elapsed_ms)

    return SearchResponse(
        results=search_results,
        query_elapsed_ms=query_elapsed_ms,
        search_elapsed_ms=search_elapsed_ms,
        total_elapsed_ms=total_elapsed_ms,
    )


@app.post("/update-dynamics", response_model=UpdateDynamicsResponse)
async def update_dynamics(req: UpdateDynamicsRequest):
    start = time.time()
    updated = 0

    point_ids = [
        make_point_id(item["memory_id"], item["ctx_index"])
        for item in req.retrieved
    ]

    try:
        points = qdrant.retrieve(
            collection_name=COLLECTION,
            ids=point_ids,
            with_vectors=True,
        )
    except Exception as e:
        logger.error(f"Failed to retrieve points for update: {e}")
        elapsed_ms = (time.time() - start) * 1000
        log_activity("update_dynamics", {"error": str(e)}, elapsed_ms)
        return UpdateDynamicsResponse(updated=0, elapsed_ms=elapsed_ms)

    points_by_id = {p.id: p for p in points}

    # 1. Reconsolidation
    for pid in point_ids:
        if pid not in points_by_id:
            continue
        payload = points_by_id[pid].payload
        qdrant.set_payload(
            collection_name=COLLECTION,
            payload={
                "last_accessed": time.time(),
                "access_count": payload.get("access_count", 0) + 1,
            },
            points=[pid],
        )
        updated += 1

    # 2. Hebbian learning
    hebbian_pairs = 0
    if len(points) > 1:
        pid_list = [p.id for p in points]

        effective = {p.id: np.array(p.vector) for p in points}
        offsets = {
            p.id: np.array(p.payload.get("offset", [0.0] * TOTAL_DIM))
            for p in points
        }

        for i, p1_id in enumerate(pid_list):
            for p2_id in pid_list[i + 1:]:
                eff1 = effective[p1_id]
                eff2 = effective[p2_id]

                offsets[p1_id] = offsets[p1_id] + req.lr * (eff2 - eff1)
                offsets[p2_id] = offsets[p2_id] + req.lr * (eff1 - eff2)

                mem1_id = points_by_id[p1_id].payload["memory_id"]
                mem2_id = points_by_id[p2_id].payload["memory_id"]

                co1 = points_by_id[p1_id].payload.get("co_occurrence", {})
                co2 = points_by_id[p2_id].payload.get("co_occurrence", {})
                co1[mem2_id] = co1.get(mem2_id, 0) + 1
                co2[mem1_id] = co2.get(mem1_id, 0) + 1

                hebbian_pairs += 1

        upsert_points = []
        for pid in pid_list:
            p = points_by_id[pid]
            base = np.array(p.payload.get("base_embedding", p.vector))
            new_effective = base + offsets[pid]

            payload = p.payload.copy()
            payload["offset"] = offsets[pid].tolist()

            upsert_points.append(PointStruct(
                id=pid,
                vector=new_effective.tolist(),
                payload=payload,
            ))

        if upsert_points:
            qdrant.upsert(
                collection_name=COLLECTION,
                points=upsert_points,
            )

    elapsed_ms = (time.time() - start) * 1000

    log_activity("update_dynamics", {
        "reconsolidated": updated,
        "hebbian_pairs": hebbian_pairs,
        "lr": req.lr,
    }, elapsed_ms)

    return UpdateDynamicsResponse(updated=updated, elapsed_ms=elapsed_ms)
