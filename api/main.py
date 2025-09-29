import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import joblib

from .endpoints import router as api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Intent Classification API", version="1.0")

# Allow CORS in Codespaces/demo scenarios
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.on_event("startup")
def load_model_on_startup():
    """
    Load model once on startup and attach to router for fast access.
    Expects model at ML/model.pkl (or env MODEL_PATH).
    """
    model_path = os.getenv("MODEL_PATH", "ML/model.pkl")
    model_file = Path(model_path)
    if not model_file.exists():
        logger.error("Model file not found at %s. API will run but /api/classify will return 503.", model_path)
        return

    try:
        model = joblib.load(model_file)
        # attach to router so endpoints can access
        api_router.model = model
        logger.info("Model loaded from %s", model_path)
    except Exception as e:
        logger.exception("Failed to load model: %s", e)
