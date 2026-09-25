"""Incremental CRF scoring for hybrid lexical-neural segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np

from wordrobe._neural_boundary import (
    MODEL_RESOURCE,
    NUM_TAGS,
    NeuralBoundaryModel,
    supports_neural_text,
)


@dataclass(frozen=True, slots=True)
class NeuralSequenceScores:
    """Precomputed CRF terms for one input run."""

    emissions: np.ndarray
    transitions: np.ndarray
    start_transitions: np.ndarray
    end_transitions: np.ndarray

    def edge_options(self, start: int, end: int, previous_tag: int) -> tuple[tuple[int, float], ...]:
        """Return ``(final_tag, score)`` options for one proposed token span.

        DKSplit label 1 marks a word start and label 0 marks continuation. At
        character zero either label represents the same segmentation, so both
        are retained until the CRF transition state makes one preferable.
        """
        seq_len = self.emissions.shape[0]
        if not 0 <= start < end <= seq_len:
            msg = f"invalid neural edge [{start}:{end}] for length {seq_len}"
            raise ValueError(msg)
        if start > 0 and not 0 <= previous_tag < NUM_TAGS:
            msg = f"invalid previous CRF tag: {previous_tag}"
            raise ValueError(msg)

        first_tags = range(NUM_TAGS) if start == 0 else (1,)
        options: list[tuple[int, float]] = []
        for first_tag in first_tags:
            if start == 0:
                score = self.start_transitions[first_tag] + self.emissions[start, first_tag]
            else:
                score = self.transitions[previous_tag, first_tag] + self.emissions[start, first_tag]

            current_tag = first_tag
            for index in range(start + 1, end):
                score += self.transitions[current_tag, 0] + self.emissions[index, 0]
                current_tag = 0

            if end == seq_len:
                score += self.end_transitions[current_tag]
            options.append((current_tag, float(score)))

        return tuple(options)


class NeuralBoundaryScorer:
    """Load the neural model and expose exact incremental CRF edge scores."""

    def __init__(self) -> None:
        self._model = NeuralBoundaryModel()
        resource = resources.files("wordrobe").joinpath(MODEL_RESOURCE)
        with resource.open("rb") as stream, np.load(stream) as data:
            self._transitions = np.asarray(data["crf_transitions"], dtype=np.float32)
            self._start_transitions = np.asarray(data["crf_start_transitions"], dtype=np.float32)
            self._end_transitions = np.asarray(data["crf_end_transitions"], dtype=np.float32)

    @staticmethod
    def supports(text: str) -> bool:
        """Return whether the neural model can score *text* without loss."""
        return supports_neural_text(text)

    def score_run(self, text: str) -> NeuralSequenceScores:
        """Compute emissions once and return reusable CRF scoring terms."""
        return NeuralSequenceScores(
            emissions=self._model.emissions(text),
            transitions=self._transitions,
            start_transitions=self._start_transitions,
            end_transitions=self._end_transitions,
        )


__all__ = ["NeuralBoundaryScorer", "NeuralSequenceScores"]
