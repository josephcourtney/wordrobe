# ruff: file-ignore: INP001
"""Compare the minimal NumPy DKSplit runtime with the published ONNX model.

This development-only script intentionally owns all ONNX Runtime and DKSplit
reference dependencies. The NumPy runtime itself imports neither package.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.request
from pathlib import Path

import dksplit
import numpy as np
from dksplit.split import Splitter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dksplit_numpy.runtime import MAX_LEN, NumpyDKSplit, text_to_ids  # ruff: ignore: E402

BENCHMARK_URL = "https://raw.githubusercontent.com/ABTdomain/dksplit/main/benchmark/sample_1000.csv"
MAX_DIVERGENCES = 20

REPRESENTATIVE_INPUTS = [
    "chatgptlogin",
    "spotifywrapped",
    "kubernetescluster",
    "openaikey",
    "microsoftoffice",
    "thisisatest",
    "isthisacamel",
    "isthisasnorql",
    "thisisaxyzzy",
    "findacme",
    "openai",
    "openaimodel",
    "scroot",
    "scrootstatus",
    "loadfrobnicate",
    "frobnicatefile",
    "parsexyzzyresponse",
    "nowhere",
    "somewhere",
    "therefore",
    "therapist",
    "expertsexchange",
    "penisland",
    "xmlhttprequest",
    "ipv6address",
    "sha256sum",
    "macOSVersion",
    "YouTubePlayer",
    "thisisnotable",
    "cannot",
    "foo-bar_baz@example.com",
    "naive.cafe-42",
]


def _installed_model_paths() -> tuple[Path, Path]:
    model_dir = Path(dksplit.__file__).resolve().parent / "models"
    return model_dir / "dksplit-int8.onnx", model_dir / "dksplit.npz"


def _reference_emissions(splitter: Splitter, text: str) -> np.ndarray:
    processed = text.lower()[:MAX_LEN]
    ids = text_to_ids(processed)
    result = splitter.session.run(None, {"chars": ids.reshape(1, -1)})[0]
    return np.asarray(result[0], dtype=np.float32)


def _load_benchmark(url: str) -> list[dict[str, str]]:
    with urllib.request.urlopen(url, timeout=30) as response:  # ruff: ignore: S310
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _acceptable(row: dict[str, str]) -> set[str]:
    return {value.strip().lower() for value in (row["truth"], row.get("might_right", "")) if value.strip()}


def _run_representative(reference: Splitter, model: NumpyDKSplit) -> None:
    top1_matches = 0
    top3_order_matches = 0
    max_error = 0.0
    total_error = 0.0
    emission_values = 0
    divergences: list[tuple[str, list[str], list[str]]] = []

    for text in REPRESENTATIVE_INPUTS:
        reference_emissions = _reference_emissions(reference, text)
        numpy_emissions = model.emissions(text)
        if reference_emissions.shape != numpy_emissions.shape:
            message = (
                f"emission shape mismatch for {text!r}: "
                f"{reference_emissions.shape} != {numpy_emissions.shape}"
            )
            raise AssertionError(message)
        difference = np.abs(reference_emissions - numpy_emissions)
        max_error = max(max_error, float(np.max(difference)))
        total_error += float(np.sum(difference))
        emission_values += difference.size

        expected = reference.split(text)
        actual = model.split(text)
        expected_top3 = reference.split_topk(text, 3)
        actual_top3 = model.split_topk(text, 3)
        top1_matches += expected == actual
        top3_order_matches += expected_top3 == actual_top3
        if expected != actual:
            divergences.append((text, expected, actual))

    count = len(REPRESENTATIVE_INPUTS)
    print(f"representative_samples={count}")
    print(f"representative_top1_parity={top1_matches / count:.6%}")
    print(f"representative_top3_order_parity={top3_order_matches / count:.6%}")
    print(f"representative_max_abs_emission_error={max_error:.9g}")
    print(f"representative_mean_abs_emission_error={total_error / emission_values:.9g}")
    for text, expected, actual in divergences:
        print(f"representative_divergence={text!r} onnx={expected!r} numpy={actual!r}")


def _run_benchmark(reference: Splitter, model: NumpyDKSplit, url: str) -> None:  # ruff: ignore: PLR0914, PLR0915
    rows = _load_benchmark(url)
    top1_parity = 0
    top3_order_parity = 0
    top5_order_parity = 0
    reference_strict = 0
    numpy_strict = 0
    reference_lenient = 0
    numpy_lenient = 0
    reference_top3 = 0
    numpy_top3 = 0
    reference_top5 = 0
    numpy_top5 = 0
    reference_only = 0
    numpy_only = 0
    max_error = 0.0
    total_error = 0.0
    emission_values = 0
    reference_seconds = 0.0
    numpy_seconds = 0.0
    divergences: list[tuple[str, str, str, set[str]]] = []

    for row in rows:
        text = row["prefix"]
        truth = row["truth"].strip().lower()
        acceptable = _acceptable(row)

        started = time.perf_counter()
        reference_emissions = _reference_emissions(reference, text)
        reference_words = reference.split(text)
        reference_candidates = reference.split_topk(text, 5)
        reference_seconds += time.perf_counter() - started

        started = time.perf_counter()
        numpy_emissions = model.emissions(text)
        numpy_words = model.split(text)
        numpy_candidates = model.split_topk(text, 5)
        numpy_seconds += time.perf_counter() - started

        difference = np.abs(reference_emissions - numpy_emissions)
        max_error = max(max_error, float(np.max(difference)))
        total_error += float(np.sum(difference))
        emission_values += difference.size

        reference_top1 = " ".join(reference_words)
        numpy_top1 = " ".join(numpy_words)
        reference_strings = [" ".join(words) for words in reference_candidates]
        numpy_strings = [" ".join(words) for words in numpy_candidates]

        top1_parity += reference_top1 == numpy_top1
        top3_order_parity += reference_strings[:3] == numpy_strings[:3]
        top5_order_parity += reference_strings == numpy_strings
        reference_strict += reference_top1 == truth
        numpy_strict += numpy_top1 == truth

        reference_ok = reference_top1 in acceptable
        numpy_ok = numpy_top1 in acceptable
        reference_lenient += reference_ok
        numpy_lenient += numpy_ok
        reference_top3 += any(candidate in acceptable for candidate in reference_strings[:3])
        numpy_top3 += any(candidate in acceptable for candidate in numpy_strings[:3])
        reference_top5 += any(candidate in acceptable for candidate in reference_strings)
        numpy_top5 += any(candidate in acceptable for candidate in numpy_strings)
        reference_only += reference_ok and not numpy_ok
        numpy_only += numpy_ok and not reference_ok

        if reference_top1 != numpy_top1 and len(divergences) < MAX_DIVERGENCES:
            divergences.append((text, reference_top1, numpy_top1, acceptable))

    count = len(rows)
    print(f"benchmark_samples={count}")
    print(f"benchmark_top1_parity={top1_parity / count:.6%}")
    print(f"benchmark_top3_order_parity={top3_order_parity / count:.6%}")
    print(f"benchmark_top5_order_parity={top5_order_parity / count:.6%}")
    print(f"benchmark_onnx_strict_top1={reference_strict / count:.6%}")
    print(f"benchmark_numpy_strict_top1={numpy_strict / count:.6%}")
    print(f"benchmark_onnx_lenient_top1={reference_lenient / count:.6%}")
    print(f"benchmark_numpy_lenient_top1={numpy_lenient / count:.6%}")
    print(f"benchmark_onnx_lenient_top3={reference_top3 / count:.6%}")
    print(f"benchmark_numpy_lenient_top3={numpy_top3 / count:.6%}")
    print(f"benchmark_onnx_lenient_top5={reference_top5 / count:.6%}")
    print(f"benchmark_numpy_lenient_top5={numpy_top5 / count:.6%}")
    print(f"benchmark_onnx_only_lenient_top1={reference_only}")
    print(f"benchmark_numpy_only_lenient_top1={numpy_only}")
    print(f"benchmark_max_abs_emission_error={max_error:.9g}")
    print(f"benchmark_mean_abs_emission_error={total_error / emission_values:.9g}")
    print(f"benchmark_onnx_seconds={reference_seconds:.6f}")
    print(f"benchmark_numpy_seconds={numpy_seconds:.6f}")
    for text, expected, actual, acceptable in divergences:
        print(
            f"benchmark_divergence={text!r} onnx={expected!r} numpy={actual!r} "
            f"acceptable={sorted(acceptable)!r}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="converted NumPy runtime NPZ")
    parser.add_argument("--benchmark-url", default=BENCHMARK_URL)
    return parser


def main() -> None:
    args = _parser().parse_args()
    onnx_path, crf_path = _installed_model_paths()
    reference = Splitter(model_path=str(onnx_path), crf_path=str(crf_path), num_threads=1)
    model = NumpyDKSplit(args.weights)
    print(f"onnx_bytes={onnx_path.stat().st_size}")
    print(f"converted_bytes={args.weights.stat().st_size}")
    _run_representative(reference, model)
    _run_benchmark(reference, model, args.benchmark_url)


if __name__ == "__main__":
    main()
