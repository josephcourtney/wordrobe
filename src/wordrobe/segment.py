"""Low-resource word segmentation with lexical and boundary evidence."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import TYPE_CHECKING, Literal, Protocol, cast

from wordrobe._segment_core import (
    COMMON_WORDS,
    SYSTEM_WORDLISTS,
    normalize_word,
)
from wordrobe._segment_core import (
    WordSegmenter as _CoreWordSegmenter,
)

if TYPE_CHECKING:
    from os import PathLike

DEFAULT_CUSTOM_WORD_COST = 5.0
DEFAULT_NUMERIC_COST = 1.0
DEFAULT_CASE_BOUNDARY_BONUS = 4.0
DEFAULT_NUMERIC_BOUNDARY_BONUS = 2.5
DEFAULT_UNKNOWN_CHAR_COST = 4.0
DEFAULT_UNKNOWN_SHORT_FRAGMENT_PENALTY = 10.0
DEFAULT_WEAK_SHORT_WORD_PENALTY = 20.0
DEFAULT_IMPLICIT_BOUNDARY_PENALTY = 5.0
DEFAULT_ADJACENT_SINGLETON_PENALTY = 24.0
DEFAULT_ARTICLE_UNKNOWN_BONUS = 6.0
DEFAULT_NEURAL_WEIGHT = 1.0

SpanKind = Literal["token", "separator"]
UnknownCost = Callable[[str], float]
ExtraWords = Mapping[str, float] | Collection[str]
_ViterbiState = tuple[bool, bool, int]  # singleton, contextual article, previous neural tag
_ViterbiCosts = list[dict[_ViterbiState, float]]
_ViterbiBack = list[dict[_ViterbiState, tuple[int, _ViterbiState]]]
_NO_NEURAL_TAG = -1
_START_STATE: _ViterbiState = (False, False, _NO_NEURAL_TAG)
_ARTICLES = frozenset({"a", "an", "the"})
_SHORT_WORD_THRESHOLD = 3
_ARTICLE_UNKNOWN_MIN_LENGTH = 5


class _NeuralScores(Protocol):
    def edge_options(self, start: int, end: int, previous_tag: int) -> tuple[tuple[int, float], ...]: ...


class _NeuralScorer(Protocol):
    def supports(self, text: str) -> bool: ...

    def score_run(self, text: str) -> _NeuralScores: ...


class BoundaryModel(StrEnum):
    """Boundary-evidence backends available to ``WordSegmenter``."""

    HEURISTIC = "heuristic"
    DKSPLIT = "dksplit"


@dataclass(frozen=True, slots=True)
class SegmentSpan:
    """A lossless token or separator span in the original input."""

    text: str
    start: int
    end: int
    kind: SpanKind


def _load_neural_scorer() -> _NeuralScorer:
    try:
        module = import_module("wordrobe._neural_scoring")
    except ImportError as exc:
        msg = "DKSplit boundary evidence requires NumPy; install wordrobe[neural]"
        raise ImportError(msg) from exc
    return cast("_NeuralScorer", module.NeuralBoundaryScorer())


class WordSegmenter(_CoreWordSegmenter):
    """Segment text using lexical costs plus case, punctuation, and local sequence evidence."""

    def __init__(
        self,
        wordlist: str | PathLike[str] | None = None,
        frequency_file: str | PathLike[str] | None = None,
        max_word_length: int = 32,
        unknown_base_cost: float = 12.0,
        unknown_char_cost: float = DEFAULT_UNKNOWN_CHAR_COST,
        *,
        extra_words: ExtraWords = (),
        blocked_words: Collection[str] = (),
        unknown_cost: UnknownCost | None = None,
        custom_word_cost: float = DEFAULT_CUSTOM_WORD_COST,
        numeric_cost: float = DEFAULT_NUMERIC_COST,
        use_case_hints: bool = True,
        case_boundary_bonus: float = DEFAULT_CASE_BOUNDARY_BONUS,
        numeric_boundary_bonus: float = DEFAULT_NUMERIC_BOUNDARY_BONUS,
        weak_short_word_penalty: float = DEFAULT_WEAK_SHORT_WORD_PENALTY,
        implicit_boundary_penalty: float = DEFAULT_IMPLICIT_BOUNDARY_PENALTY,
        adjacent_singleton_penalty: float = DEFAULT_ADJACENT_SINGLETON_PENALTY,
        article_unknown_bonus: float = DEFAULT_ARTICLE_UNKNOWN_BONUS,
        boundary_model: BoundaryModel | str = BoundaryModel.HEURISTIC,
        neural_weight: float = DEFAULT_NEURAL_WEIGHT,
    ) -> None:
        self._custom_costs = self._prepare_extra_words(extra_words, custom_word_cost)
        self._blocked_words = frozenset(filter(None, (self._normalize_custom_word(word) for word in blocked_words)))
        self._unknown_cost = unknown_cost
        self.numeric_cost = numeric_cost
        self.use_case_hints = use_case_hints
        self.case_boundary_bonus = case_boundary_bonus
        self.numeric_boundary_bonus = numeric_boundary_bonus
        self.weak_short_word_penalty = self._finite_nonnegative(weak_short_word_penalty, "weak_short_word_penalty")
        self.implicit_boundary_penalty = self._finite_nonnegative(
            implicit_boundary_penalty,
            "implicit_boundary_penalty",
        )
        self.adjacent_singleton_penalty = self._finite_nonnegative(
            adjacent_singleton_penalty,
            "adjacent_singleton_penalty",
        )
        self.article_unknown_bonus = self._finite_nonnegative(article_unknown_bonus, "article_unknown_bonus")
        self.boundary_model = BoundaryModel(boundary_model)
        self.neural_weight = self._finite_nonnegative(neural_weight, "neural_weight")
        self._neural_scorer = _load_neural_scorer() if self.boundary_model is BoundaryModel.DKSPLIT else None
        configured_max_word_length = max_word_length

        super().__init__(
            wordlist=wordlist,
            frequency_file=frequency_file,
            max_word_length=max_word_length,
            unknown_base_cost=unknown_base_cost,
            unknown_char_cost=unknown_char_cost,
        )

        # Parent initialization shortens max_word_length to the longest known
        # dictionary word. That optimization is invalid when unknown spans are
        # legal candidates: a long unknown identifier must remain representable.
        self.max_word_length = configured_max_word_length

        self.words.update(self._custom_costs)
        self.cost.update(self._custom_costs)

    @staticmethod
    def _finite_nonnegative(value: float, name: str) -> float:
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            msg = f"{name} must be a finite non-negative number"
            raise ValueError(msg)
        return numeric

    def _load_common_words(self) -> None:
        """Load ranked fallback words, preserving the best rank of duplicates."""
        n = len(COMMON_WORDS)
        normalizer = math.log(n + 1)

        for rank, word in enumerate(COMMON_WORDS, 1):
            self.words.add(word)
            self.cost.setdefault(word, math.log(rank * normalizer))

    @staticmethod
    def _normalize_custom_word(word: str) -> str | None:
        normalized = word.strip().lower().replace("'", "").replace("\u2019", "")
        return normalized if normalized and normalized.isalnum() else None

    @classmethod
    def _prepare_extra_words(cls, extra_words: ExtraWords, default_cost: float) -> dict[str, float]:
        if isinstance(extra_words, str):
            items = ((extra_words, default_cost),)
        elif isinstance(extra_words, Mapping):
            items = extra_words.items()
        else:
            items = ((word, default_cost) for word in extra_words)

        result: dict[str, float] = {}
        for word, cost in items:
            normalized = cls._normalize_custom_word(word)
            if normalized is None:
                continue
            numeric_cost = float(cost)
            if not math.isfinite(numeric_cost):
                msg = f"custom word cost must be finite: {word!r}"
                raise ValueError(msg)
            result[normalized] = numeric_cost
        return result

    def _default_unknown_cost(self, word: str) -> float:
        """Return a compositional cost for an unknown word candidate."""
        short_fragment_penalty = DEFAULT_UNKNOWN_SHORT_FRAGMENT_PENALTY * max(
            0,
            _SHORT_WORD_THRESHOLD - len(word),
        )
        return self.unknown_base_cost + self.unknown_char_cost * len(word) + short_fragment_penalty

    def _is_unknown(self, word: str) -> bool:
        return (
            not word.isdigit() and word not in self._custom_costs and word not in self.words and word not in self.cost
        )

    def _known_word_cost(self, word: str) -> float | None:
        """Return the cost of known lexical evidence, or None for an OOV candidate."""
        if word in self.cost:
            return super().word_cost(word)
        if word not in self.words:
            return None

        weak_penalty = self.weak_short_word_penalty * max(0, _SHORT_WORD_THRESHOLD - len(word))
        return super().word_cost(word) + weak_penalty

    def _unknown_word_cost(self, word: str) -> float:
        """Return configured or default cost for an OOV candidate."""
        if self._unknown_cost is None:
            return self._default_unknown_cost(word)

        cost = float(self._unknown_cost(word))
        if not math.isfinite(cost):
            msg = f"unknown-word cost must be finite: {word!r}"
            raise ValueError(msg)
        return cost

    def word_cost(self, word: str) -> float:
        """Return lexical cost, honoring custom, blocked, numeric, weak-dictionary, and unknown rules."""
        if word in self._blocked_words:
            return math.inf

        custom = self._custom_costs.get(word)
        if custom is not None:
            return custom

        if word.isdigit():
            return self.numeric_cost

        known = self._known_word_cost(word)
        if known is not None:
            return known

        return self._unknown_word_cost(word)

    def _boundary_cost(self, text: str, pos: int) -> float:
        if pos <= 0 or pos >= len(text):
            return 0.0

        cost = self.implicit_boundary_penalty
        left = text[pos - 1]
        right = text[pos]

        if left.isdigit() != right.isdigit():
            return cost - self.numeric_boundary_bonus

        if not self.use_case_hints:
            return cost

        if left.islower() and right.isupper():
            return cost - self.case_boundary_bonus

        if left.isupper() and right.isupper() and pos + 1 < len(text) and text[pos + 1].islower():
            return cost - self.case_boundary_bonus

        return cost

    def _transition_cost(self, previous: _ViterbiState, word: str) -> float:
        previous_singleton, previous_article, _ = previous
        current_singleton = len(word) == 1 and word.isalpha()
        cost = 0.0

        if previous_singleton and current_singleton:
            cost += self.adjacent_singleton_penalty

        if previous_article and len(word) >= _ARTICLE_UNKNOWN_MIN_LENGTH and self._is_unknown(word):
            cost -= self.article_unknown_bonus

        return cost

    @staticmethod
    def _state_for(word: str, *, has_prefix: bool, neural_tag: int) -> _ViterbiState:
        return (len(word) == 1 and word.isalpha(), has_prefix and word in _ARTICLES, neural_tag)

    @staticmethod
    def _neural_edge_options(
        neural_scores: _NeuralScores | None,
        start: int,
        end: int,
        previous_tag: int,
    ) -> tuple[tuple[int, float], ...]:
        if neural_scores is None:
            return ((_NO_NEURAL_TAG, 0.0),)
        return neural_scores.edge_options(start, end, previous_tag)

    def _relax_word(
        self,
        text: str,
        start: int,
        end: int,
        costs: _ViterbiCosts,
        back: _ViterbiBack,
        neural_scores: _NeuralScores | None,
    ) -> None:
        previous_costs = costs[start]
        if not previous_costs:
            return

        word = text[start:end].lower()
        lexical_cost = self.word_cost(word)
        if not math.isfinite(lexical_cost):
            return

        base_cost = lexical_cost + self._boundary_cost(text, start)
        end_costs = costs[end]
        end_back = back[end]

        for previous_state, previous_cost in previous_costs.items():
            transition_cost = self._transition_cost(previous_state, word)
            for neural_tag, neural_score in self._neural_edge_options(
                neural_scores,
                start,
                end,
                previous_state[2],
            ):
                state = self._state_for(word, has_prefix=start > 0, neural_tag=neural_tag)
                candidate = previous_cost + base_cost + transition_cost - self.neural_weight * neural_score
                if candidate >= end_costs.get(state, math.inf):
                    continue
                end_costs[state] = candidate
                end_back[state] = (start, previous_state)

    @staticmethod
    def _backtrack(text: str, costs: _ViterbiCosts, back: _ViterbiBack) -> list[str]:
        pos = len(text)
        if not costs[pos]:
            return [text]

        state = min(costs[pos], key=costs[pos].__getitem__)
        result: list[str] = []
        while pos > 0:
            previous = back[pos].get(state)
            if previous is None:
                return [text]
            start, previous_state = previous
            result.append(text[start:pos])
            pos = start
            state = previous_state

        result.reverse()
        return result

    def _neural_scores(self, text: str) -> _NeuralScores | None:
        if self._neural_scorer is None or not self._neural_scorer.supports(text):
            return None
        return self._neural_scorer.score_run(text)

    def _segment_run(self, text: str) -> list[str]:
        """Return the minimum-cost segmentation for one alphanumeric run."""
        if not text:
            return []

        n = len(text)
        neural_scores = self._neural_scores(text)
        costs: _ViterbiCosts = [{} for _ in range(n + 1)]
        back: _ViterbiBack = [{} for _ in range(n + 1)]
        costs[0][_START_STATE] = 0.0

        for end in range(1, n + 1):
            start_min = max(0, end - self.max_word_length)
            for start in range(start_min, end):
                self._relax_word(text, start, end, costs, back, neural_scores)

        return self._backtrack(text, costs, back)

    def segment_spans(self, text: str) -> list[SegmentSpan]:
        """Return lossless token/separator spans with offsets into *text*."""
        if not text:
            return []

        spans: list[SegmentSpan] = []
        run_start = 0
        run_is_word = text[0].isalnum()

        for index in range(1, len(text) + 1):
            at_end = index == len(text)
            changes_kind = not at_end and text[index].isalnum() != run_is_word
            if not at_end and not changes_kind:
                continue

            run = text[run_start:index]
            if run_is_word:
                token_start = run_start
                for token in self._segment_run(run):
                    token_end = token_start + len(token)
                    spans.append(SegmentSpan(token, token_start, token_end, "token"))
                    token_start = token_end
            else:
                spans.append(SegmentSpan(run, run_start, index, "separator"))

            if not at_end:
                run_start = index
                run_is_word = text[index].isalnum()

        return spans

    def segment(self, text: str) -> list[str]:
        """Return segmented tokens, preserving their original spelling and case."""
        return [span.text for span in self.segment_spans(text) if span.kind == "token"]

    def segment_string(self, text: str) -> str:
        return " ".join(self.segment(text))


__all__ = [
    "BoundaryModel",
    "COMMON_WORDS",
    "DEFAULT_ADJACENT_SINGLETON_PENALTY",
    "DEFAULT_ARTICLE_UNKNOWN_BONUS",
    "DEFAULT_CASE_BOUNDARY_BONUS",
    "DEFAULT_CUSTOM_WORD_COST",
    "DEFAULT_IMPLICIT_BOUNDARY_PENALTY",
    "DEFAULT_NEURAL_WEIGHT",
    "DEFAULT_NUMERIC_BOUNDARY_BONUS",
    "DEFAULT_NUMERIC_COST",
    "DEFAULT_UNKNOWN_CHAR_COST",
    "DEFAULT_UNKNOWN_SHORT_FRAGMENT_PENALTY",
    "DEFAULT_WEAK_SHORT_WORD_PENALTY",
    "SYSTEM_WORDLISTS",
    "SegmentSpan",
    "WordSegmenter",
    "normalize_word",
]
