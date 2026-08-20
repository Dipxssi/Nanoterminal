from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
import uuid


class Turn(BaseModel):
    role: Literal["user", "assistant", "tool", "system"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MemoryRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    memory_type: str
    timestamp: datetime = Field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None
    source_turn_indices: List[int] = Field(default_factory=list)