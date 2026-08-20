from typing import List, Optional
from memory.schemas import Turn


class SessionBuffer:
    def __init__(self, max_token_limit: int = 2000):
        self.turns: List[Turn] = []
        self.max_token_limit = max_token_limit

    def add_turn(
        self,
        role: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[dict] = None
    ) -> Turn:
        """Appends a new turn to the working buffer in RAM."""
        turn = Turn(
            role=role,
            content=content,
            embedding=embedding,
            metadata=metadata or {}
        )
        self.turns.append(turn)
        return turn

    def get_turns(self) -> List[Turn]:
        """Returns all recorded turns in the active buffer."""
        return self.turns

    def is_empty(self) -> bool:
        return len(self.turns) == 0

    def clear(self):
        """Resets the buffer after consolidation."""
        self.turns.clear()