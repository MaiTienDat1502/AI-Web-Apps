"""FastAPI backend cho 4 chức năng AI."""

import io
import json
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from config import (
    CORS_ORIGINS,
    DEVICE,
    ENABLED_MODELS,
    MAX_UPLOAD_MB,
    ROOT,
)

from core.llm import RAGChatbot
from core.retrieval import ImageSearch
from core.classifier import ImageClassifier
from core.detector import ObjectDetector


# =========================================================
# MODEL INSTANCES
# =========================================================

MODELS = {}

RETRIEVAL = None
CLASSIFIER = None
DETECTOR = None


# =========================================================
# LOAD MODELS
# =========================================================

def load_models():
    global RETRIEVAL
    global CLASSIFIER
    global DETECTOR

    # -------------------------
    # RAG CHATBOT
    # -------------------------
    if "llm" in ENABLED_MODELS:
        try:
            print("Loading RAG chatbot...")
            MODELS["llm"] = RAGChatbot()
            print("Loaded RAG chatbot")
        except Exception as exc:
            print(f"Cannot load RAG chatbot: {exc}")

    # -------------------------
    # IMAGE RETRIEVAL
    # -------------------------
    if "retrieval" in ENABLED_MODELS:
        try:
            print("Loading Image Retrieval...")
            RETRIEVAL = ImageSearch()
            print("Loaded Image Retrieval")
        except Exception as exc:
            print(f"Cannot load Image Retrieval: {exc}")

    # -------------------------
    # IMAGE CLASSIFICATION
    # -------------------------
    if "classifier" in ENABLED_MODELS:
        try:
            print("Loading Image Classifier...")
            CLASSIFIER = ImageClassifier()
            print("Loaded Image Classifier")
        except Exception as exc:
            print(f"Cannot load Image Classifier: {exc}")

    # -------------------------
    # OBJECT DETECTION
    # -------------------------
    if "detector" in ENABLED_MODELS:
        try:
            print("Loading Object Detector...")
            DETECTOR = ObjectDetector()
            print("Loaded Object Detector")
        except Exception as exc:
            print(f"Cannot load Object Detector: {exc}")


# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()

    yield

    MODELS.clear()

    global RETRIEVAL
    global CLASSIFIER
    global DETECTOR

    RETRIEVAL = None
    CLASSIFIER = None
    DETECTOR = None


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AI Web Apps API",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST TIMING
# =========================================================

@app.middleware("http")
async def timing(request, call_next):
    t0 = time.perf_counter()

    response = await call_next(request)

    elapsed = (time.perf_counter() - t0) * 1000

    response.headers["X-Process-Time-ms"] = f"{elapsed:.1f}"

    return response


# =========================================================
# MODEL CHECK FUNCTIONS
# =========================================================

def require_llm():
    if "llm" not in MODELS:
        raise HTTPException(
            status_code=503,
            detail="Mô hình RAG chưa được nạp.",
        )

    return MODELS["llm"]


def require_retrieval():
    if RETRIEVAL is None:
        raise HTTPException(
            status_code=503,
            detail="Mô hình Image Retrieval chưa được nạp.",
        )

    return RETRIEVAL


def require_classifier():
    if CLASSIFIER is None:
        raise HTTPException(
            status_code=503,
            detail="Mô hình Image Classification chưa được nạp.",
        )

    return CLASSIFIER


def require_detector():
    if DETECTOR is None:
        raise HTTPException(
            status_code=503,
            detail="Mô hình Object Detection chưa được nạp.",
        )

    return DETECTOR


# =========================================================
# IMAGE READING
# =========================================================

async def read_image(file: UploadFile) -> Image.Image:
    data = await file.read()

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Ảnh vượt quá {MAX_UPLOAD_MB} MB.",
        )

    try:
        image = Image.open(io.BytesIO(data))

        image.load()

        return image.convert("RGB")

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="File không phải ảnh hợp lệ.",
        )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "models": {
            "llm": "llm" in MODELS,
            "retrieval": RETRIEVAL is not None,
            "classifier": CLASSIFIER is not None,
            "detector": DETECTOR is not None,
        },
    }


# =========================================================
# 1. IMAGE CLASSIFICATION
# =========================================================

@app.post("/api/classify")
async def classify(
    file: UploadFile = File(...),
    top_k: int = Form(3),
):
    classifier = require_classifier()

    image = await read_image(file)

    if top_k < 1:
        top_k = 1

    if top_k > 5:
        top_k = 5

    result = classifier.predict(
        image,
        top_k=top_k,
    )

    return {
        "filename": file.filename,
        "predictions": result["predictions"],
        "confident": result["confident"],
    }


# =========================================================
# 2. OBJECT DETECTION
# =========================================================

@app.post("/api/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
):
    detector = require_detector()

    image = await read_image(file)

    if confidence < 0.05:
        confidence = 0.05

    if confidence > 0.95:
        confidence = 0.95

    result = detector.predict(
        image,
        confidence=confidence,
    )

    # Lấy ảnh đã được YOLO vẽ bounding box
    annotated_image = result["annotated_image"]

    # Encode ảnh thành JPEG
    output = io.BytesIO()

    annotated_image.save(
        output,
        format="JPEG",
        quality=90,
    )

    output.seek(0)

    import base64

    image_base64 = base64.b64encode(
        output.read()
    ).decode("utf-8")

    return {
        "filename": file.filename,
        "count": result["count"],
        "detections": result["detections"],
        "image": f"data:image/jpeg;base64,{image_base64}",
    }


# =========================================================
# 3. IMAGE RETRIEVAL - TEXT
# =========================================================

@app.post("/api/search/text")
def search_text(
    query: str = Form(...),
    k: int = Form(8),
):
    retrieval = require_retrieval()

    query = query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query không được để trống.",
        )

    if k < 1:
        k = 1

    results = retrieval.search_text(
        query,
        k=k,
    )

    return {
        "query": query,
        "results": results,
    }


# =========================================================
# 4. IMAGE RETRIEVAL - IMAGE
# =========================================================

@app.post("/api/search/image")
async def search_image(
    file: UploadFile = File(...),
    k: int = Form(8),
):
    retrieval = require_retrieval()

    image = await read_image(file)

    if k < 1:
        k = 1

    results = retrieval.search_image(
        image,
        k=k,
    )

    return {
        "filename": file.filename,
        "results": results,
    }


# =========================================================
# RAG CHATBOT REQUEST
# =========================================================

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    history: list[dict] = Field(
        default_factory=list,
    )


# =========================================================
# RAG CHATBOT - STREAMING
# =========================================================

@app.post("/api/chat")
def chat(req: ChatRequest):
    bot = require_llm()

    contexts, tokens = bot.stream(
        req.message,
        req.history,
    )

    def events():

        # Gửi sources trước
        sources_data = {
            "type": "sources",
            "items": contexts,
        }

        yield (
            "data: "
            + json.dumps(
                sources_data,
                ensure_ascii=False,
            )
            + "\n\n"
        )

        # Gửi từng token
        for piece in tokens:

            token_data = {
                "type": "token",
                "text": piece,
            }

            yield (
                "data: "
                + json.dumps(
                    token_data,
                    ensure_ascii=False,
                )
                + "\n\n"
            )

        # Kết thúc
        yield 'data: {"type":"done"}\n\n'

    return StreamingResponse(
        events(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
        },
    )


# =========================================================
# RAG CHATBOT - SYNC
# =========================================================

@app.post("/api/chat/sync")
def chat_sync(req: ChatRequest):
    bot = require_llm()

    return bot.answer(
        req.message,
        req.history,
    )


# =========================================================
# STATIC IMAGE DIRECTORY
# =========================================================

IMAGE_DIR = ROOT / "data" / "images"

if IMAGE_DIR.exists():

    app.mount(
        "/images",
        StaticFiles(
            directory=str(IMAGE_DIR)
        ),
        name="images",
    )


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return {
        "name": "AI Web Apps API",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/classify",
            "/api/detect",
            "/api/search/text",
            "/api/search/image",
            "/api/chat",
            "/api/chat/sync",
        ],
    }