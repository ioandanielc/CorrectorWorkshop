"""
src/inference/correctors/base.py
-----------------
Shared interfaces so multiple corrector/experiment implementations can
be swapped without touching their callers.
"""
from abc import ABC, abstractmethod

import numpy as np


class Corrector(ABC):
    """A point-cloud corrector: k passes over (N, 2) points -> (N, 2) points."""

    @abstractmethod
    def apply(self, points: np.ndarray, k: int = 1) -> np.ndarray:
        ...


class Experiment(ABC):
    """A runnable experiment: builds its own state from a config, then runs."""

    @abstractmethod
    def run(self) -> None:
        ...
