import uuid

from pydantic import BaseModel, Field


class TextInput(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(min_length=1, max_length=10_000)


class LabelResult(BaseModel):
    prob: float
    flagged: bool


class ModerationResult(BaseModel):
    id: str
    text: str
    toxicity: LabelResult
    hate: LabelResult
    safe: bool
    processing_time_ms: float


class BatchTextRequest(BaseModel):
    items: list[TextInput] = Field(min_length=1, max_length=100)


class BatchTextResponse(BaseModel):
    items: list[ModerationResult]
    total_processing_time_ms: float
