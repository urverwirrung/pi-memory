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


# --- Helpers ---

def make_point_id(memory_id: str, ctx_index: int) -> str:
    """Deterministic UUID from memory_id:ctx_index for Qdrant."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{memory_id}:{ctx_index}"))


def compute_strength(initial_strength: float, access_count: int,
                     decay_rate: float, last_accessed: float) -> float:
    """Temporal strength: decays over time, boosted by access."""
    age = time.time() - last_accessed
    return (
        initial_strength
        * (1 + math.log(max(access_count, 1)))
        * math.exp(-decay_rate * age)
    )


# --- Request/Response Models ---

class EmbedRequest(BaseModel):
    """Structured input for embedding.

    The six state dimensions:
    - certainty: how well do I understand what's going on?
    - clarity: how clear is what we're optimizing for?
    - scope: how broad vs deep should attention be?
    - stakes: what are the consequences of getting this wrong?
    - valence: how well are things going? (positive = aligned, negative = misaligned)
    - arousal: how significant is this moment? (high = activated, low = routine)

    The first four (eigenvectors) describe cognitive posture toward the task.
    The last two describe cognitive posture toward my own performance —
    the elephant's signal to the monkey. Self-assessed, not computed.
    """
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
    """Store a memory node with its embedding."""
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
    co_occurrence: dict[str, int] = Field(default_factory=dict)


class StoreResponse(BaseModel):
    point_id: str
    elapsed_ms: float


class SearchRequest(BaseModel):
    """Search for relevant memories."""
    query: EmbedRequest
    top_k: int = 10
    exclude_memory_ids: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    memory_id: str
    ctx_index: int
    content: str
    context_summary: str
    score: float
    initial_strength: float
    decay_rate: float
    last_accessed: float
    access_count: int
    co_occurrence: dict[str, int]


class SearchResponse(BaseModel):
    results: list[SearchResult]
    elapsed_ms: float


class UpdateDynamicsRequest(BaseModel):
    """Update temporal dynamics and co-occurrence after retrieval."""
    retrieved: list[dict]  # [{memory_id, ctx_index}]
    lr: float = 0.01


class UpdateDynamicsResponse(BaseModel):
    updated: int
    elapsed_ms: float


# --- Model Loading ---

def load_model():
    """Load GTE-Qwen2-7B-instruct with int8 quantization."""
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
    """Connect to Qdrant and ensure collection exists."""
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

def format_structured_input(req: EmbedRequest) -> str:
    """Format structured input for the embedding model."""
    parts = [f"[CONTENT: {req.content}]"]
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
    """Compute embedding: model output (3584d) + state dimensions (6d) = 3590d."""
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

    # Last hidden state, mean pooling over non-padding tokens
    attention_mask = inputs["attention_mask"]
    hidden = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
    sum_embeddings = torch.sum(hidden * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_pooled = (sum_embeddings / sum_mask).squeeze(0)

    # Normalize
    mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=0)

    # Convert to list and append state dimensions
    base_vec = mean_pooled.cpu().float().numpy().tolist()
    return base_vec + state_dims


def embed_request(req: EmbedRequest) -> list[float]:
    """Full pipeline: format structured input → compute embedding."""
    text = format_structured_input(req)
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
    return {
        "status": "ok",
        "model": os.environ.get("MODEL_NAME", "unknown"),
        "collection": COLLECTION,
        "total_dim": TOTAL_DIM,
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    """Embed a single structured input."""
    start = time.time()
    vector = embed_request(req)
    elapsed_ms = (time.time() - start) * 1000
    return EmbedResponse(vector=vector, elapsed_ms=elapsed_ms)


@app.post("/embed/batch", response_model=BatchEmbedResponse)
async def embed_batch(req: BatchEmbedRequest):
    """Embed multiple structured inputs."""
    start = time.time()
    vectors = [embed_request(r) for r in req.inputs]
    elapsed_ms = (time.time() - start) * 1000
    return BatchEmbedResponse(vectors=vectors, elapsed_ms=elapsed_ms)


@app.post("/store", response_model=StoreResponse)
async def store(req: StoreRequest):
    """Embed and store a memory node in Qdrant."""
    start = time.time()

    vector = embed_request(req.embed)
    point_id = make_point_id(req.memory_id, req.ctx_index)

    # Store the base embedding in payload (immutable historical record)
    # The Qdrant vector = base + offset (effective embedding for search)
    # Initially offset is zero, so vector = base
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
    return StoreResponse(point_id=point_id, elapsed_ms=elapsed_ms)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """
    Embed query and search for relevant memories.

    Search uses effective embeddings (base + offset) by updating vectors
    in-place after Hebbian learning. The vectors in Qdrant already reflect
    cumulative offsets.
    """
    start = time.time()

    query_vector = embed_request(req.query)

    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=req.top_k * 3 if req.exclude_memory_ids else req.top_k,
    )

    search_results = []
    seen_memory_ids = set()

    for hit in results:
        payload = hit.payload
        mem_id = payload["memory_id"]

        if mem_id in req.exclude_memory_ids:
            continue
        if mem_id in seen_memory_ids:
            continue
        seen_memory_ids.add(mem_id)

        strength = compute_strength(
            payload.get("initial_strength", 1.0),
            payload.get("access_count", 0),
            payload.get("decay_rate", 0.05),
            payload.get("last_accessed", time.time()),
        )
        weighted_score = hit.score * strength

        search_results.append(SearchResult(
            memory_id=mem_id,
            ctx_index=payload.get("ctx_index", 0),
            content=payload.get("content", ""),
            context_summary=payload.get("context_summary", ""),
            score=weighted_score,
            initial_strength=payload.get("initial_strength", 1.0),
            decay_rate=payload.get("decay_rate", 0.05),
            last_accessed=payload.get("last_accessed", 0),
            access_count=payload.get("access_count", 0),
            co_occurrence=payload.get("co_occurrence", {}),
        ))

    search_results.sort(key=lambda r: r.score, reverse=True)
    search_results = search_results[:req.top_k]

    elapsed_ms = (time.time() - start) * 1000
    return SearchResponse(results=search_results, elapsed_ms=elapsed_ms)


@app.post("/update-dynamics", response_model=UpdateDynamicsResponse)
async def update_dynamics(req: UpdateDynamicsRequest):
    """
    Post-retrieval updates: reconsolidation + Hebbian learning.

    Hebbian learning updates the OFFSET, not the base embedding.
    The base embedding is the immutable historical record (including
    original valence/arousal at encoding time). The offset accumulates
    drift from co-occurrence. The Qdrant vector = base + offset, so
    search always uses effective embeddings.

    The delta between base and effective embedding on the valence/arousal
    dimensions IS the therapeutic change — how my relationship to a
    memory has evolved through re-experiencing it in new contexts.
    """
    start = time.time()
    updated = 0

    # Gather all retrieved points
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
        return UpdateDynamicsResponse(updated=0, elapsed_ms=elapsed_ms)

    points_by_id = {p.id: p for p in points}

    # 1. Reconsolidation — update access time and count
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

    # 2. Hebbian learning — co-retrieved pairs drift closer via offsets
    if len(points) > 1:
        pid_list = [p.id for p in points]

        # Get effective embeddings (current vectors) and offsets
        effective = {p.id: np.array(p.vector) for p in points}
        offsets = {
            p.id: np.array(p.payload.get("offset", [0.0] * TOTAL_DIM))
            for p in points
        }

        for i, p1_id in enumerate(pid_list):
            for p2_id in pid_list[i + 1:]:
                eff1 = effective[p1_id]
                eff2 = effective[p2_id]

                # Update offsets — drift toward each other
                offsets[p1_id] = offsets[p1_id] + req.lr * (eff2 - eff1)
                offsets[p2_id] = offsets[p2_id] + req.lr * (eff1 - eff2)

                # Update co-occurrence counts
                mem1_id = points_by_id[p1_id].payload["memory_id"]
                mem2_id = points_by_id[p2_id].payload["memory_id"]

                co1 = points_by_id[p1_id].payload.get("co_occurrence", {})
                co2 = points_by_id[p2_id].payload.get("co_occurrence", {})
                co1[mem2_id] = co1.get(mem2_id, 0) + 1
                co2[mem1_id] = co2.get(mem1_id, 0) + 1

        # Recompute effective vectors (base + updated offset) and write back
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
    return UpdateDynamicsResponse(updated=updated, elapsed_ms=elapsed_ms)
