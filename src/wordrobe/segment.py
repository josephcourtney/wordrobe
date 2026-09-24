"""Low-resource word segmentation with lexical and boundary evidence."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

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
DEFAULT_UNKNOWN_CHAR_COST = 2.0

SpanKind = Literal["token", "separator"]
UnknownCost = Callable[[str], float]
ExtraWords = Mapping[str, float] | Collection[str]


@dataclass(frozen=True, slots=True)
class SegmentSpan:
    """A lossless token or separator span in the original input."""

    text: str
    start: int
    end: int
    kind: SpanKind


class WordSegmenter(_CoreWordSegmenter):
    """Segment text using lexical costs plus case and punctuation boundaries."""

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
    ) -> None:
        self._custom_costs = self._prepare_extra_words(extra_words, custom_word_cost)
        self._blocked_words = frozenset(filter(None, (self._normalize_custom_word(word) for word in blocked_words)))
        self._unknown_cost = unknown_cost
        self.numeric_cost = numeric_cost
        self.use_case_hints = use_case_hints
        self.case_boundary_bonus = case_boundary_bonus
        self.numeric_boundary_bonus = numeric_boundary_bonus
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
        # A linear character term is essential here. The former sqrt(length)
        # term made one long unknown span systematically cheaper than several
        # ordinary known words, defeating Viterbi segmentation on inputs such
        # as ``thisisatest``. A small short-fragment penalty also discourages
        # artifacts such as splitting an unknown word into ``came`` + ``l``.
        short_fragment_penalty = max(0, 3 - len(word))
        return self.unknown_base_cost + self.unknown_char_cost * len(word) + short_fragment_penalty

    def word_cost(self, word: str) -> float:
        """Return lexical cost, honoring custom, blocked, numeric, and unknown rules."""
        if word in self._blocked_words:
            return math.inf

        custom = self._custom_costs.get(word)
        if custom is not None:
            return custom

        if word.isdigit():
            return self.numeric_cost

        if word in self.words or word in self.cost:
            return super().word_cost(word)

        if self._unknown_cost is not None:
            cost = float(self._unknown_cost(word))
            if not math.isfinite(cost):
                msg = f"unknown-word cost must be finite: {word!r}"
                raise ValueError(msg)
            return cost

        return self._default_unknown_cost(word)

    def _boundary_cost(self, text: str, pos: int) -> float:
        if pos <= 0 or pos >= len(text):
            return 0.0

        left = text[pos - 1]
        right = text[pos]

        if left.isdigit() != right.isdigit():
            return -self.numeric_boundary_bonus

        if not self.use_case_hints:
            return 0.0

        if left.islower() and right.isupper():
            return -self.case_boundary_bonus

        if left.isupper() and right.isupper() and pos + 1 < len(text) and text[pos + 1].islower():
            return -self.case_boundary_bonus

        return 0.0

    def _segment_run(self, text: str) -> list[str]:
        """Return the minimum-cost segmentation for one alphanumeric run."""
        if not text:
            return []

        n = len(text)
        dp = [math.inf] * (n + 1)
        back = [-1] * (n + 1)
        dp[0] = 0.0

        for end in range(1, n + 1):
            start_min = max(0, end - self.max_word_length)
            for start in range(start_min, end):
                word = text[start:end].lower()
                candidate = dp[start] + self.word_cost(word) + self._boundary_cost(text, start)
                if candidate < dp[end]:
                    dp[end] = candidate
                    back[end] = start

        result: list[str] = []
        pos = n
        while pos > 0:
            start = back[pos]
            if start < 0:
                return [text]
            result.append(text[start:pos])
            pos = start

        result.reverse()
        return result

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
    "COMMON_WORDS",
    "DEFAULT_CASE_BOUNDARY_BONUS",
    "DEFAULT_CUSTOM_WORD_COST",
    "DEFAULT_NUMERIC_BOUNDARY_BONUS",
    "DEFAULT_NUMERIC_COST",
    "DEFAULT_UNKNOWN_CHAR_COST",
    "SYSTEM_WORDLISTS",
    "SegmentSpan",
    "WordSegmenter",
    "normalize_word",
]
