from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from wordrobe._neural_boundary import (
    MAX_LEN,
    NeuralBoundaryModel,
    UnsupportedNeuralTextError,
    supports_neural_text,
)


@pytest.fixture(scope="module")
def model() -> NeuralBoundaryModel:
    return NeuralBoundaryModel()


@pytest.mark.parametrize(
    ("text", "supported"),
    [
        ("chatgptlogin", True),
        ("ipv6address", True),
        ("a" * MAX_LEN, True),
        ("", False),
        ("foo-bar", False),
        ("café", False),
        ("a" * (MAX_LEN + 1), False),
    ],
)
def test_supports_neural_text(text: str, *, supported: bool) -> None:
    assert supports_neural_text(text) is supported


def test_emissions_have_two_scores_per_character(model: NeuralBoundaryModel) -> None:
    emissions = model.emissions("chatgptlogin")

    assert emissions.shape == (len("chatgptlogin"), 2)
    assert emissions.dtype == np.float32
    assert np.isfinite(emissions).all()


def test_best_labels_are_first_topk_path(model: NeuralBoundaryModel) -> None:
    text = "expertsexchange"

    best = model.best_labels(text)
    candidates = model.topk_labels(text, 5)

    assert candidates[0] == best
    assert len(best) == len(text)
    assert all(label in {0, 1} for label in best)
    assert len(set(candidates)) == len(candidates)


def test_topk_paths_are_sorted_by_crf_score(model: NeuralBoundaryModel) -> None:
    text = "thisisnotable"
    candidates = model.topk_labels(text, 5)
    scores = [model.score_labels(text, labels) for labels in candidates]

    assert scores == sorted(scores, reverse=True)


def test_model_is_case_insensitive(model: NeuralBoundaryModel) -> None:
    assert model.best_labels("macOSVersion") == model.best_labels("macosversion")


@pytest.mark.parametrize("text", ["", "foo-bar", "café", "a" * (MAX_LEN + 1)])
def test_inference_rejects_unsupported_inputs(model: NeuralBoundaryModel, text: str) -> None:
    with pytest.raises(UnsupportedNeuralTextError):
        model.emissions(text)


def test_score_labels_validates_shape_and_values(model: NeuralBoundaryModel) -> None:
    with pytest.raises(ValueError, match="expected 4 labels"):
        model.score_labels("test", [0, 1])
    with pytest.raises(ValueError, match="labels must be"):
        model.score_labels("test", [0, 0, 0, 2])


def test_topk_requires_positive_k(model: NeuralBoundaryModel) -> None:
    with pytest.raises(ValueError, match="k must be >= 1"):
        model.topk_labels("test", 0)
