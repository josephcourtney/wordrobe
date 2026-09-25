from __future__ import annotations

import math

import pytest

from wordrobe.segment import DEFAULT_NEURAL_WEIGHT, BoundaryModel, WordSegmenter


@pytest.fixture
def empty_wordlist(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")
    return path


def test_default_neural_weight_is_calibrated() -> None:
    assert math.isclose(DEFAULT_NEURAL_WEIGHT, 0.5)


@pytest.mark.parametrize(
    "text",
    [
        "thisisatest",
        "isthisacamel",
        "findacme",
        "expertsexchange",
        "ipv6address",
    ],
)
def test_zero_neural_weight_matches_heuristic(empty_wordlist, text: str) -> None:
    heuristic = WordSegmenter(wordlist=empty_wordlist)
    neural = WordSegmenter(
        wordlist=empty_wordlist,
        boundary_model=BoundaryModel.DKSPLIT,
        neural_weight=0.0,
    )

    assert neural.segment(text) == heuristic.segment(text)


def test_neural_evidence_can_change_the_viterbi_winner(empty_wordlist) -> None:
    heuristic = WordSegmenter(wordlist=empty_wordlist)
    neural = WordSegmenter(
        wordlist=empty_wordlist,
        boundary_model=BoundaryModel.DKSPLIT,
        neural_weight=3.0,
    )

    assert heuristic.segment("therapist") == ["the", "rapist"]
    assert neural.segment("therapist") == ["therapist"]


def test_default_neural_weight_preserves_article_oov_regression(empty_wordlist) -> None:
    segmenter = WordSegmenter(wordlist=empty_wordlist, boundary_model=BoundaryModel.DKSPLIT)

    assert segmenter.segment("isthisasnorql") == ["is", "this", "a", "snorql"]


@pytest.mark.parametrize(
    "text",
    [
        "caféthisistext",
        "thisisatest" * 6,
    ],
)
def test_neural_backend_falls_back_for_unsupported_runs(empty_wordlist, text: str) -> None:
    heuristic = WordSegmenter(wordlist=empty_wordlist)
    neural = WordSegmenter(wordlist=empty_wordlist, boundary_model=BoundaryModel.DKSPLIT)

    assert neural.segment(text) == heuristic.segment(text)


def test_neural_backend_preserves_custom_word_evidence(empty_wordlist) -> None:
    segmenter = WordSegmenter(
        wordlist=empty_wordlist,
        extra_words={"scroot": 0.0},
        boundary_model=BoundaryModel.DKSPLIT,
    )

    assert segmenter.segment("scroot") == ["scroot"]


def test_neural_backend_preserves_blocked_words(empty_wordlist) -> None:
    segmenter = WordSegmenter(
        wordlist=empty_wordlist,
        blocked_words={"therapist"},
        boundary_model=BoundaryModel.DKSPLIT,
    )

    assert "therapist" not in [word.lower() for word in segmenter.segment("therapist")]


def test_boundary_model_accepts_string_value(empty_wordlist) -> None:
    segmenter = WordSegmenter(wordlist=empty_wordlist, boundary_model="dksplit")

    assert segmenter.boundary_model is BoundaryModel.DKSPLIT


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_neural_weight_must_be_finite_nonnegative(empty_wordlist, value: float) -> None:
    with pytest.raises(ValueError, match="neural_weight"):
        WordSegmenter(wordlist=empty_wordlist, neural_weight=value)
