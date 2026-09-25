"""Evaluate hybrid lexical-neural segmentation over adversarial Wordrobe cases."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from wordrobe.segment import BoundaryModel, WordSegmenter

CASES: tuple[tuple[str, frozenset[tuple[str, ...]]], ...] = (
    ("thisisatest", frozenset({("this", "is", "a", "test")})),
    ("isthisacamel", frozenset({("is", "this", "a", "camel")})),
    ("isthisasnorql", frozenset({("is", "this", "a", "snorql")})),
    ("thisisaxyzzy", frozenset({("this", "is", "a", "xyzzy")})),
    ("findacme", frozenset({("find", "acme")})),
    ("makeanobject", frozenset({("make", "an", "object")})),
    ("theoryworks", frozenset({("theory", "works")})),
    ("anotherthing", frozenset({("another", "thing")})),
    ("openai", frozenset({("openai",)})),
    ("openaimodel", frozenset({("openai", "model")})),
    ("scroot", frozenset({("scroot",)})),
    ("scrootstatus", frozenset({("scroot", "status")})),
    ("loadfrobnicate", frozenset({("load", "frobnicate")})),
    ("frobnicatefile", frozenset({("frobnicate", "file")})),
    ("parsexyzzyresponse", frozenset({("parse", "xyzzy", "response")})),
    ("nowhere", frozenset({("nowhere",)})),
    ("somewhere", frozenset({("somewhere",)})),
    ("therefore", frozenset({("therefore",)})),
    ("therapist", frozenset({("therapist",)})),
    ("expertsexchange", frozenset({("experts", "exchange")})),
    ("penisland", frozenset({("pen", "island")})),
    ("xmlhttprequest", frozenset({("xml", "http", "request")})),
    ("ipv6address", frozenset({("ipv6", "address")})),
    ("sha256sum", frozenset({("sha256", "sum")})),
    ("thisisnotable", frozenset({("this", "is", "notable"), ("this", "is", "not", "able")})),
    ("cannot", frozenset({("cannot",), ("can", "not")})),
)

EXTRA_WORDS = {
    "openai": 1.0,
    "scroot": 1.0,
    "load": 1.0,
    "find": 1.0,
    "ipv6": 1.0,
    "sha256": 1.0,
    "xml": 1.0,
    "http": 1.0,
}
DEFAULT_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def evaluate(weight: float, wordlist: Path) -> tuple[int, list[tuple[str, tuple[str, ...]]]]:
    segmenter = WordSegmenter(
        wordlist=wordlist,
        extra_words=EXTRA_WORDS,
        boundary_model=BoundaryModel.DKSPLIT,
        neural_weight=weight,
    )
    mismatches: list[tuple[str, tuple[str, ...]]] = []
    for text, accepted in CASES:
        actual = tuple(word.lower() for word in segmenter.segment(text))
        if actual not in accepted:
            mismatches.append((text, actual))
    return len(CASES) - len(mismatches), mismatches


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="*", type=float, default=DEFAULT_WEIGHTS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    with TemporaryDirectory() as directory:
        wordlist = Path(directory) / "empty.txt"
        wordlist.write_text("", encoding="utf-8")
        for weight in args.weights:
            correct, mismatches = evaluate(weight, wordlist)
            print(f"weight={weight:g} correct={correct}/{len(CASES)}")
            for text, actual in mismatches:
                print(f"  {text} -> {' '.join(actual)}")


if __name__ == "__main__":
    main()
