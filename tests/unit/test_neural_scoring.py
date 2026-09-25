from __future__ import annotations

import pytest

from wordrobe._neural_boundary import NeuralBoundaryModel
from wordrobe._neural_scoring import NeuralBoundaryScorer


def _labels(length: int, boundaries: set[int], first_tag: int) -> tuple[int, ...]:
    return tuple(first_tag if index == 0 else int(index in boundaries) for index in range(length))


def _edge_totals(text: str, boundaries: tuple[int, ...]) -> list[float]:
    scores = NeuralBoundaryScorer().score_run(text)
    starts = (0, *boundaries)
    ends = (*boundaries, len(text))
    paths: list[tuple[int, float]] = [(-1, 0.0)]
    for start, end in zip(starts, ends, strict=True):
        next_paths: list[tuple[int, float]] = []
        for previous_tag, accumulated in paths:
            for final_tag, edge_score in scores.edge_options(start, end, previous_tag):
                next_paths.append((final_tag, accumulated + edge_score))
        paths = next_paths
    return sorted(score for _, score in paths)


def test_incremental_edges_equal_complete_crf_scores() -> None:
    text = "thisisnotable"
    boundaries = (4, 6)
    model = NeuralBoundaryModel()

    complete = sorted(model.score_labels(text, _labels(len(text), set(boundaries), first_tag)) for first_tag in (0, 1))

    assert _edge_totals(text, boundaries) == pytest.approx(complete)


def test_neural_edge_scoring_rejects_invalid_ranges() -> None:
    scores = NeuralBoundaryScorer().score_run("test")

    with pytest.raises(ValueError, match="invalid neural edge"):
        scores.edge_options(2, 2, 0)
    with pytest.raises(ValueError, match="invalid previous CRF tag"):
        scores.edge_options(1, 4, -1)
