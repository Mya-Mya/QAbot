from dataclasses import dataclass
from typing import List

@dataclass
class Answer:
    score: float
    start: int
    end: int