from abc import ABC, abstractmethod
from typing import List
from answer import Answer


class QABot(ABC):
    @abstractmethod
    def set_context(self, context: str):
        pass

    @abstractmethod
    def extract_context(self, start: int, end: int) -> str:
        pass

    @abstractmethod
    def ask_question(self, question: str, topk: int) -> List[Answer]:
        pass
