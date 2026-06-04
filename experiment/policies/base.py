from __future__ import annotations

from abc import ABC, abstractmethod

from experiment.context import DocumentContext


class PolicyBase(ABC):
    @abstractmethod
    def select(self, context: DocumentContext, rng) -> list:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
