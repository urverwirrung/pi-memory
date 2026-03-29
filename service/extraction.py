"""
Pi Memory — Extraction & Utility Generation Service

Runs Qwen2.5-7B-Instruct on the 5080 for:
- Memory extraction: identifies discrete memories from exchanges
- Summarization: compresses context for various purposes
- Utility generation: any lightweight generation task
"""

import os
import time
import json
import logging
from collections import deque
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pi-extraction")

# --- Globals ---
model = None
tokenizer = None

MAX_ACTIVITY_LOG = 100
activity_log = deque(maxlen=MAX_ACTIVITY_LOG)


def log_activity(operation: str, details: dict, elapsed_ms: float):
    entry = {
        "timestamp": time.time(),
        "operation": operation,
        "elapsed_ms": round(elapsed_ms, 2),
        **details,
    }
    activity_log.appendleft(entry)
    logger.info(f"[{operation}] {elapsed_ms:.1f}ms — {details}")


# --- Request/Response Models ---

class ExtractRequest(BaseModel):
    user_message: str
    assistant_response: str
    context: str = ""
    max_memories: int = 5


class ExtractedMemory(BaseModel):
    content: str
    context_summary: str
    type: str = "episodic"


class ExtractResponse(BaseModel):
    memories: list[ExtractedMemory]
    raw_output: str           # raw model output for debugging
    elapsed_ms: float


class SummarizeRequest(BaseModel):
    text: str
    instruction: str = "Summarize the key points concisely."
    max_tokens: int = 500


class SummarizeResponse(BaseModel):
    summary: str
    elapsed_ms: float


class GenerateRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    max_tokens: int = 1000
    temperature: float = 0.3


class GenerateResponse(BaseModel):
    text: str
    elapsed_ms: float


# --- Model Loading ---

def load_model():
    global model, tokenizer

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
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
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map=device,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map=device,
        )

    model.eval()
    elapsed = time.time() - start
    logger.info(f"Model loaded in {elapsed:.1f}s")


def generate(messages: list[dict], max_tokens: int = 500, temperature: float = 0.3) -> str:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    logger.info("Pi Extraction service ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Pi Extraction Service",
    description="Memory extraction and utility generation for Pi's cognitive infrastructure",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": os.environ.get("MODEL_NAME", "unknown"),
        "service": "extraction",
    }


@app.get("/activity")
async def get_activity(limit: int = 20):
    return {"operations": list(activity_log)[:limit]}


EXTRACTION_SYSTEM = """You extract discrete memories from conversation exchanges.

Given a user message and assistant response, identify what is worth remembering.
Focus on:
- Decisions made and their reasoning
- Insights or realizations
- Patterns noticed
- Preferences expressed (by either party)
- Important facts or context established
- Open questions or tensions identified

Skip:
- Routine procedural content (tool calls, file operations)
- Boilerplate and ceremony
- Content that is purely ephemeral

Return a JSON array of memories. Each memory has:
- "content": the memory text (concise, self-contained, 1-3 sentences)
- "context_summary": brief context of when/why this emerged (1 sentence)
- "type": one of "decision", "insight", "pattern", "preference", "fact", "question"

Return ONLY the JSON array, no other text. If nothing is worth remembering, return []."""


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    start = time.time()

    context_line = f"\nConversation context: {req.context}" if req.context else ""

    messages = [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": f"""Extract memories from this exchange:{context_line}

USER: {req.user_message}

ASSISTANT: {req.assistant_response}"""},
    ]

    raw = generate(messages, max_tokens=1000, temperature=0.1)

    memories = []
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if "```" in cleaned:
                cleaned = cleaned[:cleaned.rindex("```")]
            cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            for item in parsed[:req.max_memories]:
                if isinstance(item, dict) and "content" in item:
                    memories.append(ExtractedMemory(
                        content=item["content"],
                        context_summary=item.get("context_summary", ""),
                        type=item.get("type", "episodic"),
                    ))
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse extraction output: {e}")
        logger.debug(f"Raw output: {raw}")

    elapsed_ms = (time.time() - start) * 1000

    log_activity("extract", {
        "user_preview": req.user_message[:80],
        "memories_extracted": len(memories),
        "memory_types": [m.type for m in memories],
    }, elapsed_ms)

    return ExtractResponse(memories=memories, raw_output=raw, elapsed_ms=elapsed_ms)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(req: SummarizeRequest):
    start = time.time()

    messages = [
        {"role": "system", "content": "You are a precise summarizer. Be concise and preserve key information."},
        {"role": "user", "content": f"{req.instruction}\n\n{req.text}"},
    ]

    summary = generate(messages, max_tokens=req.max_tokens, temperature=0.2)

    elapsed_ms = (time.time() - start) * 1000

    log_activity("summarize", {
        "input_length": len(req.text),
        "output_length": len(summary),
    }, elapsed_ms)

    return SummarizeResponse(summary=summary.strip(), elapsed_ms=elapsed_ms)


@app.post("/generate", response_model=GenerateResponse)
async def gen(req: GenerateRequest):
    start = time.time()

    messages = []
    if req.system_prompt:
        messages.append({"role": "system", "content": req.system_prompt})
    messages.append({"role": "user", "content": req.prompt})

    text = generate(messages, max_tokens=req.max_tokens, temperature=req.temperature)

    elapsed_ms = (time.time() - start) * 1000

    log_activity("generate", {
        "prompt_preview": req.prompt[:80],
        "output_length": len(text),
    }, elapsed_ms)

    return GenerateResponse(text=text.strip(), elapsed_ms=elapsed_ms)
