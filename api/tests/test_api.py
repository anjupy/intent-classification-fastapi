import os
import json
from fastapi.testclient import TestClient
import joblib
from pathlib import Path
import pytest

# Import app
from api.main import app

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def ensure_model_loaded():
    """
    Ensure ML/model.pkl exists; if not, create a very small dummy pipeline for tests.
    This avoids failing tests if ML/model.pkl is missing in fresh clones.
    """
    model_path = Path("ML/model.pkl")
    if not model_path.exists():
        # create dummy sklearn pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        dummy = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression())
        ])
        # fit dummy model on minimal data
        X = ["hello", "book a table"]
        y = ["greeting", "book"]
        dummy.fit(X, y)
        joblib.dump(dummy, model_path)
    # ensure environment for auth
    os.environ.setdefault("API_ADMIN_USER", "admin")
    os.environ.setdefault("API_ADMIN_PASS", "password")
    yield
    # no teardown


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_single_classify():
    payload = {"text": "What's the weather today?"}
    r = client.post("/api/classify", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "intent" in body and "confidence" in body


def test_batch_classify():
    payload = {"texts": ["Book me a restaurant", "Hello"]}
    r = client.post("/api/classify/batch", json=payload)
    assert r.status_code == 200
    items = r.json()
    assert isinstance(items, list)
    assert len(items) == 2
    assert all("intent" in it and "confidence" in it for it in items)


def test_model_info_requires_auth():
    # without auth -> 401 or 500 if env not set
    r = client.get("/api/model/info")
    assert r.status_code in (401, 500)
