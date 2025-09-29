from typing import List, Optional
from pydantic import BaseModel


class SingleTextRequest(BaseModel):
    text: str


class SingleTextResponse(BaseModel):
    intent: str
    confidence: float


class BatchTextRequest(BaseModel):
    texts: List[str]


class BatchItem(BaseModel):
    text: str
    intent: str
    confidence: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    n_classes: int
    classes: List[str]
    performance_report: Optional[str] = None
    notes: Optional[str] = None
