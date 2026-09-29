"""FastAPI backend cho AI Web Apps."""

import base64
import importlib
import io
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import CORS_ORIGINS, DEVICE, ENABLED_MODELS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

log = logging.getLogger("api")

MODELS: dict = {}


LOADERS = {
    "classifier": ("core.classifier", "ImageClassifier"),
    "detector": ("core.detector", "ObjectDetector"),
    "retrieval": ("core.retrieval", "ImageSearch"),
    "llm": ("core.llm", "RAGChatbot"),
}


def _load_models():
    for name, (module, cls) in LOADERS.items():
        if name not in ENABLED_MODELS:
            continue

        t0 = time.perf_counter()

        try:
            MODELS[name] = getattr(
                importlib.import_module(module),
                cls
            )()

            log.info(
                "loaded %s in %.1fs",
                name,
                time.perf_counter() - t0
            )

        except Exception as exc:
            log.exception(
                "cannot load %s: %s",
                name,
                exc
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    MODELS.clear()


app = FastAPI(
    title="AI Web Apps API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()

    response = await call_next(request)

    response.headers["X-Process-Time-ms"] = (
        f"{(time.perf_counter() - t0) * 1000:.1f}"
    )

    return response


def _require(name: str):
    if name not in MODELS:
        raise HTTPException(
            503,
            f"Mô hình '{name}' chưa được nạp"
        )

    return MODELS[name]


# =========================
# HEALTH
# =========================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "models": {
            m: m in MODELS
            for m in sorted(ENABLED_MODELS)
        }
    }


# =========================
# RAG CHATBOT
# =========================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000
    )

    history: list[dict] = Field(
        default_factory=list
    )


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    Server-Sent Events:
    sources -> token -> ... -> done
    """

    bot = _require("llm")

    contexts, tokens = bot.stream(
        req.message,
        req.history
    )

    def events():
        sources_data = {
            "type": "sources",
            "items": contexts
        }

        yield f"data: {json.dumps(sources_data, ensure_ascii=False)}\n\n"

        for piece in tokens:
            token_data = {
                "type": "token",
                "text": piece
            }

            yield f"data: {json.dumps(token_data, ensure_ascii=False)}\n\n"

        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(
        events(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache"
        }
    )


@app.post("/api/chat/sync")
def chat_sync(req: ChatRequest):
    return _require("llm").answer(
        req.message,
        req.history
    )