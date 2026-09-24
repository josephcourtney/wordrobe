"""Wordrobe: a costume change for words."""

from __future__ import annotations

from wordrobe._meta import metadata
from wordrobe.case import (
    AmbiguousCaseError,
    Case,
    CaseError,
    InvalidCaseError,
    decode,
    encode,
    guess_case,
    is_reversible,
    possible_cases,
    translate,
)

__version__ = metadata.version

__all__ = [
    "AmbiguousCaseError",
    "Case",
    "CaseError",
    "InvalidCaseError",
    "__version__",
    "decode",
    "encode",
    "guess_case",
    "is_reversible",
    "possible_cases",
    "translate",
]
