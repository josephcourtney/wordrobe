from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class Case(StrEnum):
    """Supported text/identifier case conventions."""

    # Reversibly decodable because word boundaries are explicit.
    SNAKE = "snake_case"
    SCREAMING_SNAKE = "SCREAMING_SNAKE_CASE"

    KEBAB = "kebab-case"
    SCREAMING_KEBAB = "SCREAMING-KEBAB-CASE"
    TRAIN = "Train-Case"

    DOT = "dot.case"
    PATH = "path/case"

    LOWER = "lower case"
    UPPER = "UPPER CASE"
    TITLE = "Title Case"
    SENTENCE = "Sentence case"

    # Encodable, but not generally reversibly decodable.
    CAMEL = "camelCase"
    PASCAL = "PascalCase"
    FLAT = "flatcase"
    UPPER_FLAT = "UPPERFLATCASE"


class CaseError(ValueError):
    """Base class for case-related errors."""


class InvalidCaseError(CaseError):
    """Raised when text does not match any supported case."""

    def __init__(self, text: str):
        self.text = text
        super().__init__(f"{text!r} does not match any supported case.")


class AmbiguousCaseError(CaseError):
    """Raised when text matches more than one supported case."""

    def __init__(self, text: str, candidates: list[Case]):
        self.text = text
        self.candidates = candidates
        super().__init__(f"{text!r} is ambiguous: " + ", ".join(case.value for case in candidates))


# Delimiter-based cases can be decoded without guessing word boundaries.
_SEPARATORS: dict[Case, str] = {
    Case.SNAKE: "_",
    Case.SCREAMING_SNAKE: "_",
    Case.KEBAB: "-",
    Case.SCREAMING_KEBAB: "-",
    Case.TRAIN: "-",
    Case.DOT: ".",
    Case.PATH: "/",
    Case.LOWER: " ",
    Case.UPPER: " ",
    Case.TITLE: " ",
    Case.SENTENCE: " ",
}

REVERSIBLE_CASES = frozenset(_SEPARATORS)

# Deterministic display order for possible_cases().
# This is a preference/order convention, not a probability model.
CASE_PREFERENCE: tuple[Case, ...] = (
    Case.SNAKE,
    Case.SCREAMING_SNAKE,
    Case.KEBAB,
    Case.SCREAMING_KEBAB,
    Case.TRAIN,
    Case.DOT,
    Case.PATH,
    Case.LOWER,
    Case.UPPER,
    Case.TITLE,
    Case.SENTENCE,
    Case.CAMEL,
    Case.PASCAL,
    Case.FLAT,
    Case.UPPER_FLAT,
)

_ALNUM_RE = re.compile(r"^[A-Za-z0-9]+$")


def is_reversible(case: Case) -> bool:
    """Return whether *case* can be decoded without guessing boundaries."""
    return case in REVERSIBLE_CASES


def _normalize_words(words: Iterable[str]) -> list[str]:
    """
    Normalize semantic words to lowercase.

    Words are restricted to non-empty ASCII alphanumeric tokens so that
    separators remain reserved exclusively for word boundaries.
    """
    normalized: list[str] = []

    for word in words:
        if not isinstance(word, str):
            msg = f"Words must be strings, got {type(word).__name__}."
            raise TypeError(msg)

        if not word:
            msg = "Words must be non-empty."
            raise ValueError(msg)

        if _ALNUM_RE.fullmatch(word) is None:
            msg = f"Word {word!r} is not ASCII alphanumeric; component words may contain only A-Z, a-z, and 0-9."
            raise ValueError(msg)

        normalized.append(word.lower())

    return normalized


def _capitalize(word: str) -> str:
    """Capitalize the first character without lowercasing the remainder."""
    return word[:1].upper() + word[1:]


_ENCODING_STYLES = {
    Case.SNAKE: ("_", str.lower, str.lower),
    Case.SCREAMING_SNAKE: ("_", str.upper, str.upper),
    Case.KEBAB: ("-", str.lower, str.lower),
    Case.SCREAMING_KEBAB: ("-", str.upper, str.upper),
    Case.TRAIN: ("-", _capitalize, _capitalize),
    Case.DOT: (".", str.lower, str.lower),
    Case.PATH: ("/", str.lower, str.lower),
    Case.LOWER: (" ", str.lower, str.lower),
    Case.UPPER: (" ", str.upper, str.upper),
    Case.TITLE: (" ", _capitalize, _capitalize),
    Case.SENTENCE: (" ", _capitalize, str.lower),
    Case.CAMEL: ("", str.lower, _capitalize),
    Case.PASCAL: ("", _capitalize, _capitalize),
    Case.FLAT: ("", str.lower, str.lower),
    Case.UPPER_FLAT: ("", str.upper, str.upper),
}


def encode(words: Iterable[str], case: Case) -> str:
    """
    Encode semantic component words in *case*.

    Component words are normalized to lowercase before formatting.

    Examples
    --------
    >>> encode(["hello", "world"], Case.SNAKE)
    'hello_world'
    >>> encode(["hello", "world"], Case.CAMEL)
    'helloWorld'
    >>> encode(["api", "client"], Case.PASCAL)
    'ApiClient'
    """
    normalized = _normalize_words(words)
    if not normalized:
        return ""

    try:
        separator, first_transform, rest_transform = _ENCODING_STYLES[case]
    except KeyError as exc:
        msg = f"Unsupported case: {case!r}"
        raise ValueError(msg) from exc

    first, *rest = normalized
    encoded_words = [first_transform(first), *(rest_transform(word) for word in rest)]
    return separator.join(encoded_words)


def decode(text: str, case: Case) -> list[str]:
    """
    Losslessly decode a canonical, reversible case into lowercase words.

    Non-reversible cases such as camelCase, PascalCase, and flatcase are
    rejected rather than heuristically split.

    The input must be canonical for the requested case. For example,
    ``"Hello_World"`` is not valid ``snake_case``.

    Examples
    --------
    >>> decode("hello_world", Case.SNAKE)
    ['hello', 'world']
    >>> decode("HELLO-WORLD", Case.SCREAMING_KEBAB)
    ['hello', 'world']
    """
    if not isinstance(text, str):
        msg = f"text must be str, got {type(text).__name__}."
        raise TypeError(msg)

    if not is_reversible(case):
        msg = f"{case.value!r} is not reversibly decodable; its word boundaries are not explicitly encoded."
        raise CaseError(msg)

    if text == "":
        return []

    separator = _SEPARATORS[case]
    raw_words = text.split(separator)

    if any(not word for word in raw_words):
        raise InvalidCaseError(text)

    try:
        words = _normalize_words(raw_words)
    except (TypeError, ValueError) as exc:
        raise InvalidCaseError(text) from exc

    # Canonical round-trip validation also verifies capitalization.
    if encode(words, case) != text:
        raise InvalidCaseError(text)

    return words


def translate(text: str, from_case: Case, to_case: Case) -> str:
    """
    Translate canonical text from one case to another.

    The source case must be reversible. The destination may be any
    supported case.

    Examples
    --------
    >>> translate("hello_world", Case.SNAKE, Case.CAMEL)
    'helloWorld'
    """
    return encode(decode(text, from_case), to_case)


def _matches_nonreversible_case(text: str, case: Case) -> bool:
    """
    Return whether text is syntactically compatible with a non-reversible case.

    This determines possibility, not a unique parse.
    """
    if not text or _ALNUM_RE.fullmatch(text) is None:
        return False

    first = text[0]
    matches = {
        Case.CAMEL: first.islower() or first.isdigit(),
        Case.PASCAL: first.isupper() or first.isdigit(),
        Case.FLAT: text == text.lower(),
        Case.UPPER_FLAT: text == text.upper(),
    }
    return matches.get(case, False)


def possible_cases(text: str) -> list[Case]:
    """
    Return every case that *text* could canonically represent.

    The result is deterministic and follows CASE_PREFERENCE. That ordering
    is a documented preference only; it is not a statistical likelihood.

    For reversible cases, matching is strict and based on decode/encode
    round-tripping. Non-reversible cases are matched syntactically.

    Examples
    --------
    >>> possible_cases("hello_world")
    [<Case.SNAKE: 'snake_case'>]

    A single word is inherently ambiguous:

    >>> Case.SNAKE in possible_cases("hello")
    True
    >>> Case.CAMEL in possible_cases("hello")
    True
    """
    if not isinstance(text, str):
        msg = f"text must be str, got {type(text).__name__}."
        raise TypeError(msg)

    if text == "":
        # Empty text is the encoding of an empty word list in every case.
        return list(CASE_PREFERENCE)

    matches: set[Case] = set()

    for case in REVERSIBLE_CASES:
        try:
            decode(text, case)
        except CaseError:
            pass
        else:
            matches.add(case)

    for case in (Case.CAMEL, Case.PASCAL, Case.FLAT, Case.UPPER_FLAT):
        if _matches_nonreversible_case(text, case):
            matches.add(case)

    return [case for case in CASE_PREFERENCE if case in matches]


def guess_case(
    text: str,
    *,
    should_raise: bool = False,
) -> Case | None:
    """
    Return the case only when *text* has exactly one possible interpretation.

    If the text is ambiguous or invalid, return None by default.

    With ``should_raise=True``:
      * InvalidCaseError is raised when no case matches.
      * AmbiguousCaseError is raised when multiple cases match.

    Examples
    --------
    >>> guess_case("hello_world")
    <Case.SNAKE: 'snake_case'>
    >>> guess_case("hello") is None
    True
    """
    candidates = possible_cases(text)

    if len(candidates) == 1:
        return candidates[0]

    if not should_raise:
        return None

    if not candidates:
        raise InvalidCaseError(text)

    raise AmbiguousCaseError(text, candidates)


__all__ = [
    "CASE_PREFERENCE",
    "REVERSIBLE_CASES",
    "AmbiguousCaseError",
    "Case",
    "CaseError",
    "InvalidCaseError",
    "decode",
    "encode",
    "guess_case",
    "is_reversible",
    "possible_cases",
    "translate",
]
