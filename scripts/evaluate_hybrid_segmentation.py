"""Evaluate hybrid segmentation without trading away established behavior."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from wordrobe.segment import BoundaryModel, WordSegmenter


@dataclass(frozen=True, slots=True)
class Case:
    text: str
    accepted: frozenset[tuple[str, ...]]
    extra_words: tuple[tuple[str, float], ...] = ()
    xfail: bool = False

    def make_segmenter(self, wordlist: Path, weight: float) -> WordSegmenter:
        return WordSegmenter(
            wordlist=wordlist,
            extra_words=dict(self.extra_words),
            boundary_model=BoundaryModel.DKSPLIT,
            neural_weight=weight,
        )


# Mirrors test_adversarial_segmentation_cases exactly. Cases currently marked
# xfail are tracked separately: fixing them is useful, but never at the cost of
# regressing a case that the heuristic backend already guarantees.
CASES = (
    Case("thisisatest", frozenset({("this", "is", "a", "test")})),
    Case("isthisacamel", frozenset({("is", "this", "a", "camel")})),
    Case("isthisasnorql", frozenset({("is", "this", "a", "snorql")})),
    Case("thisisaxyzzy", frozenset({("this", "is", "a", "xyzzy")})),
    Case("findacme", frozenset({("find", "acme")}), (("find", 1.0),)),
    Case("openai", frozenset({("openai",)})),
    Case("scroot", frozenset({("scroot",)})),
    Case("loadfrobnicate", frozenset({("load", "frobnicate")}), (("load", 1.0),)),
    Case("frobnicatefile", frozenset({("frobnicate", "file")}), (("file", 1.0),)),
    Case(
        "parsexyzzyresponse",
        frozenset({("parse", "xyzzy", "response")}),
        (("parse", 1.0), ("response", 1.0)),
    ),
    Case("nowhere", frozenset({("nowhere",)}), xfail=True),
    Case("somewhere", frozenset({("somewhere",)}), xfail=True),
    Case("already", frozenset({("already",)})),
    Case("together", frozenset({("together",)})),
    Case("therefore", frozenset({("therefore",)}), xfail=True),
    Case("someone", frozenset({("someone",)})),
    Case("without", frozenset({("without",)})),
    Case("within", frozenset({("within",)})),
    Case("theresult", frozenset({("the", "result")})),
    Case("thereisnoway", frozenset({("there", "is", "no", "way")})),
    Case("htmlparser", frozenset({("html", "parser")}), (("html", 1.0), ("parser", 1.0))),
    Case(
        "xmlhttprequest",
        frozenset({("xml", "http", "request")}),
        (("xml", 1.0), ("http", 1.0), ("request", 1.0)),
    ),
    Case("ipv6address", frozenset({("ipv6", "address")}), (("ipv6", 1.0), ("address", 1.0))),
    Case("sha256sum", frozenset({("sha256", "sum")}), (("sha256", 1.0), ("sum", 1.0))),
    Case(
        "macOSVersion",
        frozenset({("macOS", "Version")}),
        (("macos", 1.0), ("version", 1.0)),
    ),
    Case(
        "YouTubePlayer",
        frozenset({("YouTube", "Player")}),
        (("youtube", 1.0), ("player", 1.0)),
    ),
    Case(
        "thisisnotable",
        frozenset({("this", "is", "notable"), ("this", "is", "not", "able")}),
    ),
    Case("cannot", frozenset({("cannot",), ("can", "not")})),
    Case("therapist", frozenset({("therapist",)}), xfail=True),
    Case("expertsexchange", frozenset({("experts", "exchange")}), xfail=True),
    Case("penisland", frozenset({("pen", "island")}), xfail=True),
)

DEFAULT_WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)


def _normalized(tokens: list[str]) -> tuple[str, ...]:
    return tuple(token.lower() for token in tokens)


def evaluate(
    weight: float,
    wordlist: Path,
) -> tuple[list[tuple[Case, tuple[str, ...]]], list[tuple[Case, tuple[str, ...]]]]:
    stable_failures: list[tuple[Case, tuple[str, ...]]] = []
    xfail_failures: list[tuple[Case, tuple[str, ...]]] = []
    for case in CASES:
        actual = _normalized(case.make_segmenter(wordlist, weight).segment(case.text))
        if actual in case.accepted:
            continue
        target = xfail_failures if case.xfail else stable_failures
        target.append((case, actual))
    return stable_failures, xfail_failures


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", nargs="*", type=float, default=DEFAULT_WEIGHTS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    stable_total = sum(not case.xfail for case in CASES)
    xfail_total = len(CASES) - stable_total
    with TemporaryDirectory() as directory:
        wordlist = Path(directory) / "empty.txt"
        wordlist.write_text("", encoding="utf-8")
        for weight in args.weights:
            stable_failures, xfail_failures = evaluate(weight, wordlist)
            stable_correct = stable_total - len(stable_failures)
            xfail_fixed = xfail_total - len(xfail_failures)
            print(
                f"weight={weight:g} stable={stable_correct}/{stable_total} "
                f"xfail_fixed={xfail_fixed}/{xfail_total}"
            )
            for case, actual in stable_failures:
                print(f"  REGRESSION {case.text} -> {' '.join(actual)}")
            for case, actual in xfail_failures:
                print(f"  xfail {case.text} -> {' '.join(actual)}")


if __name__ == "__main__":
    main()
