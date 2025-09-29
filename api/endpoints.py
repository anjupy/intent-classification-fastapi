import os
import json
from pathlib import Path
from typing import List, Tuple

from fastapi import APIRouter, HTTPException, status, Depends
from sklearn.base import BaseEstimator
import numpy as np

from .models import SingleTextRequest, SingleTextResponse, BatchTextRequest, BatchItem, ModelInfoResponse
from .auth import get_admin_credentials

router = APIRouter()


def _softmax(scores: np.ndarray) -> np.ndarray:
    exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _predict_with_confidence(model: BaseEstimator, texts: List[str]) -> Tuple[List[str], List[float]]:
    """
    Return (preds, confidences) for the given texts using predict_proba or decision_function fallback.
    """
    # If pipeline supports predict_proba
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
        preds_idx = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(preds_idx)), preds_idx].tolist()
        preds = model.classes_[preds_idx] if hasattr(model, "classes_") else model.predict(texts)
        return preds.tolist(), confidences

    # If classifier has decision_function (e.g., LinearSVC)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        # If binary, make 2D
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        probs = _softmax(scores)
        preds_idx = np.argmax(probs, axis=1)
        confidences = probs[np.arange(len(preds_idx)), preds_idx].tolist()
        preds = model.classes_[preds_idx] if hasattr(model, "classes_") else model.predict(texts)
        return preds.tolist(), confidences

    # Fallback: deterministic predict with confidence 1.0
    preds = model.predict(texts)
    confidences = [1.0] * len(preds)
    return preds.tolist(), confidences


# Health check
@router.get("/api/health")
def health():
    return {"status": "ok"}


# Single classify
@router.post("/api/classify", response_model=SingleTextResponse)
def classify_single(req: SingleTextRequest):
    model = getattr(router, "model", None)
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    try:
        preds, confs = _predict_with_confidence(model, [req.text])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return SingleTextResponse(intent=preds[0], confidence=float(confs[0]))


# Batch classify
@router.post("/api/classify/batch", response_model=List[BatchItem])
def classify_batch(req: BatchTextRequest):
    model = getattr(router, "model", None)
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    texts = req.texts
    if not isinstance(texts, list) or len(texts) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="`texts` must be a non-empty list")

    try:
        preds, confs = _predict_with_confidence(model, texts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    items = [BatchItem(text=t, intent=p, confidence=float(c)) for t, p, c in zip(texts, preds, confs)]
    return items


# Model info - protected with basic auth
@router.get("/api/model/info", response_model=ModelInfoResponse, dependencies=[Depends(get_admin_credentials)])
def model_info():
    model = getattr(router, "model", None)
    if model is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded")

    meta = {
        "model_name": os.getenv("MODEL_NAME", "intent-classifier"),
        "model_type": type(model.named_steps["clf"]).__name__ if hasattr(model, "named_steps") else type(model).__name__,
        "n_classes": len(model.named_steps["clf"].classes_) if hasattr(model, "named_steps") and hasattr(model.named_steps["clf"], "classes_") else -1,
        "classes": model.named_steps["clf"].classes_.tolist() if hasattr(model, "named_steps") and hasattr(model.named_steps["clf"], "classes_") else [],
        "performance_report": None,
        "notes": None
    }

    # try to read classification report from disk if present
    report_path = Path("ml/data/classification_report.txt")
    if report_path.exists():
        meta["performance_report"] = report_path.read_text(encoding="utf-8")

    # additional notes
    meta["notes"] = "Protected endpoint. Provide basic auth credentials via API_ADMIN_USER/API_ADMIN_PASS environment variables."

    return ModelInfoResponse(**meta)
