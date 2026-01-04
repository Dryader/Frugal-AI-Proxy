import uuid
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field

class LLMLog(SQLModel, table=True):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    prompt: str
    response: str
    model_used: str
    cache_hit: bool
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    savings_usd: float = 0.0
    latency_ms: int
    routing_justification: str
    shadow_response: Optional[str] = None
    shadow_model: Optional[str] = None
