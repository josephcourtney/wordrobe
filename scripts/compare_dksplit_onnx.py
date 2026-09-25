"""Compare Wordrobe's NumPy boundary model with DKSplit's published ONNX model."""

from __future__ import annotations

import argparse
import csv
import io
import time
import urllib.request
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

import numpy as np

from wordrobe._neural_boundary import CHAR_VOCAB, MAX_LEN, NeuralBoundaryModel

BENCHMARK_URL = "https://raw.githubusercontent.com/ABTdomain/dksplit/main/benchmark/sample_1000.csv"
MAX_DIVERGENCES = 20
UNK_IDX = 1

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
    "thisisnotable",
    "cannot",
]

_CHAR_MAP = np.full(128, UNK_IDX, dtype=np.int64)
for _index, _char in enumerate(CHAR_VOCAB, start=2):
    _CHAR_MAP[ord(_char)] = _index


class _ReferenceSession(Protocol):
    def run(self, output_names: object, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


class _ReferenceSplitter(Protocol):
    session: _ReferenceSession

    def split(self, text: str) -> list[str]: ...

    def split_topk(self, text: str, k: int) -> list[list[str]]: ...


def _module_path(module: ModuleType, name: str) -> Path:
    module_file = module.__file__
    if module_file is None:
        msg = f"installed {name} module has no filesystem path"
        raise RuntimeError(msg)
    return Path(module_file).resolve()


def _installed_model_paths() -> tuple[Path, Path]:
    dksplit = import_module("dksplit")
    model_dir = _module_path(dksplit, "dksplit").parent / "models"
    return model_dir / "dksplit-int8.onnx", model_dir / "dksplit.npz"


def _reference_splitter(onnx_path: Path, crf_path: Path) -> _ReferenceSplitter:
    splitter_module = import_module("dksplit.split")
    factory = cast("Callable[..., _ReferenceSplitter]", getattr(splitter_module, "Splitter"))
    return factory(model_path=str(onnx_path), crf_path=str(crf_path), num_threads=1)


def _ids(text: str) -> np.ndarray:
    raw = np.frombuffer(text.lower()[:MAX_LEN].encode("ascii"), dtype=np.uint8)
    return _CHAR_MAP[raw]


def _reference_emissions(splitter: _ReferenceSplitter, text: str) -> np.ndarray:
    result = splitter.session.run(None, {"chars": _ids(text).reshape(1, -1)})[0]
    return np.asarray(result[0], dtype=np.float32)


def _labels_to_words(text: str, labels: tuple[int, ...]) -> list[str]:
    words: list[str] = []
    current: list[str] = []
    for char, label in zip(text.lower(), labels, strict=True):
        if label == 1 and current:
            words.append("".join(current))
            current = [char]
        else:
            current.append(char)
    if current:
        words.append("".join(current))
    return words


def _model_split(model: NeuralBoundaryModel, text: str) -> list[str]:
    return _labels_to_words(text, model.best_labels(text))


def _model_topk(model: NeuralBoundaryModel, text: str, k: int) -> list[list[str]]:
    results: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for labels in model.topk_labels(text, 2 * k):
        words = _labels_to_words(text, labels)
        key = tuple(words)
        if key in seen:
            continue
        seen.add(key)
        results.append(words)
        if len(results) == k:
            break
    return results


def _load_benchmark(url: str) -> list[dict[str, str]]:
    if not url.startswith("https://"):
        msg = "benchmark URL must use https://"
        raise ValueError(msg)
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _acceptable(row: dict[str, str]) -> set[str]:
    return {value.strip().lower() for value in (row["truth"], row.get("might_right", "")) if value.strip()}


def _run_representative(reference: _ReferenceSplitter, model: NeuralBoundaryModel) -> None:
    top1_matches = 0
    top3_order_matches = 0
    max_error = 0.0
    total_error = 0.0
    emission_values = 0

    for text in REPRESENTATIVE_INPUTS:
        reference_emissions = _reference_emissions(reference, text)
        numpy_emissions = model.emissions(text)
        difference = np.abs(reference_emissions - numpy_emissions)
        max_error = max(max_error, float(np.max(difference)))
        total_error += float(np.sum(difference))
        emission_values += difference.size

        top1_matches += reference.split(text) == _model_split(model, text)
        top3_order_matches += reference.split_topk(text, 3) == _model_topk(model, text, 3)

    count = len(REPRESENTATIVE_INPUTS)
    print(f"representative_samples={count}")
    print(f"representative_top1_parity={top1_matches / count:.6%}")
    print(f"representative_top3_order_parity={top3_order_matches / count:.6%}")
    print(f"representative_max_abs_emission_error={max_error:.9g}")
    print(f"representative_mean_abs_emission_error={total_error / emission_values:.9g}")


def _run_benchmark(reference: _ReferenceSplitter, model: NeuralBoundaryModel, url: str) -> None:
    rows = _load_benchmark(url)
    top1_parity = 0
    top3_order_parity = 0
    top5_order_parity = 0
    reference_strict = 0
    model_strict = 0
    reference_lenient = 0
    model_lenient = 0
    reference_top3 = 0
    model_top3 = 0
    reference_top5 = 0
    model_top5 = 0
    max_error = 0.0
    total_error = 0.0
    emission_values = 0
    reference_seconds = 0.0
    model_seconds = 0.0
    divergences: list[tuple[str, str, str, set[str]]] = []

    for row in rows:
        text = row["prefix"]
        if not text.isascii() or not text.isalnum() or len(text) > MAX_LEN:
            continue
        truth = row["truth"].strip().lower()
        acceptable = _acceptable(row)

        started = time.perf_counter()
        reference_emissions = _reference_emissions(reference, text)
        reference_words = reference.split(text)
        reference_candidates = reference.split_topk(text, 5)
        reference_seconds += time.perf_counter() - started

        started = time.perf_counter()
        numpy_emissions = model.emissions(text)
        model_words = _model_split(model, text)
        model_candidates = _model_topk(model, text, 5)
        model_seconds += time.perf_counter() - started

        difference = np.abs(reference_emissions - numpy_emissions)
        max_error = max(max_error, float(np.max(difference)))
        total_error += float(np.sum(difference))
        emission_values += difference.size

        reference_top1 = " ".join(reference_words)
        model_top1 = " ".join(model_words)
        reference_strings = [" ".join(words) for words in reference_candidates]
        model_strings = [" ".join(words) for words in model_candidates]

        top1_parity += reference_top1 == model_top1
        top3_order_parity += reference_strings[:3] == model_strings[:3]
        top5_order_parity += reference_strings == model_strings
        reference_strict += reference_top1 == truth
        model_strict += model_top1 == truth
        reference_lenient += reference_top1 in acceptable
        model_lenient += model_top1 in acceptable
        reference_top3 += any(candidate in acceptable for candidate in reference_strings[:3])
        model_top3 += any(candidate in acceptable for candidate in model_strings[:3])
        reference_top5 += any(candidate in acceptable for candidate in reference_strings)
        model_top5 += any(candidate in acceptable for candidate in model_strings)

        if reference_top1 != model_top1 and len(divergences) < MAX_DIVERGENCES:
            divergences.append((text, reference_top1, model_top1, acceptable))

    count = sum(
        1 for row in rows if row["prefix"].isascii() and row["prefix"].isalnum() and len(row["prefix"]) <= MAX_LEN
    )
    print(f"benchmark_samples={count}")
    print(f"benchmark_top1_parity={top1_parity / count:.6%}")
    print(f"benchmark_top3_order_parity={top3_order_parity / count:.6%}")
    print(f"benchmark_top5_order_parity={top5_order_parity / count:.6%}")
    print(f"benchmark_onnx_strict_top1={reference_strict / count:.6%}")
    print(f"benchmark_numpy_strict_top1={model_strict / count:.6%}")
    print(f"benchmark_onnx_lenient_top1={reference_lenient / count:.6%}")
    print(f"benchmark_numpy_lenient_top1={model_lenient / count:.6%}")
    print(f"benchmark_onnx_lenient_top3={reference_top3 / count:.6%}")
    print(f"benchmark_numpy_lenient_top3={model_top3 / count:.6%}")
    print(f"benchmark_onnx_lenient_top5={reference_top5 / count:.6%}")
    print(f"benchmark_numpy_lenient_top5={model_top5 / count:.6%}")
    print(f"benchmark_max_abs_emission_error={max_error:.9g}")
    print(f"benchmark_mean_abs_emission_error={total_error / emission_values:.9g}")
    print(f"benchmark_onnx_seconds={reference_seconds:.6f}")
    print(f"benchmark_numpy_seconds={model_seconds:.6f}")
    for text, expected, actual, acceptable in divergences:
        print(f"benchmark_divergence={text!r} onnx={expected!r} numpy={actual!r} acceptable={sorted(acceptable)!r}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="converted Wordrobe model NPZ")
    parser.add_argument("--benchmark-url", default=BENCHMARK_URL)
    return parser


def main() -> None:
    args = _parser().parse_args()
    onnx_path, crf_path = _installed_model_paths()
    reference = _reference_splitter(onnx_path, crf_path)
    model = NeuralBoundaryModel(args.weights)
    print(f"onnx_bytes={onnx_path.stat().st_size}")
    print(f"converted_bytes={args.weights.stat().st_size}")
    _run_representative(reference, model)
    _run_benchmark(reference, model, args.benchmark_url)


if __name__ == "__main__":
    main()
