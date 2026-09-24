"""Compare float32 NumPy DKSplit with ONNX on DKSplit's published benchmark sample."""

from __future__ import annotations

import csv
import io
import tempfile
import time
import urllib.request
from pathlib import Path

import dksplit
import numpy as np
from dksplit.split import Splitter, _crf_decode_topk, _decode_predictions_batch, _text_to_ids_fast

from convert import convert_model
from float_model import FloatNumpyDKSplit
from model import MAX_LEN

BENCHMARK_URL = "https://raw.githubusercontent.com/ABTdomain/dksplit/main/benchmark/sample_1000.csv"


def _model_paths() -> tuple[Path, Path]:
    package_dir = Path(dksplit.__file__).resolve().parent
    model_dir = package_dir / "models"
    return model_dir / "dksplit-int8.onnx", model_dir / "dksplit.npz"


def _load_rows() -> list[dict[str, str]]:
    with urllib.request.urlopen(BENCHMARK_URL, timeout=30) as response:  # noqa: S310 - fixed HTTPS URL
        text = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _reference_emissions(splitter: Splitter, text: str) -> np.ndarray:
    processed = text.lower()[:MAX_LEN]
    ids = _text_to_ids_fast(processed)
    return np.asarray(splitter.session.run(None, {"chars": ids.reshape(1, -1)})[0][0], dtype=np.float32)


def _topk_words(
    text: str,
    emissions: np.ndarray,
    transitions: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    k: int,
) -> list[list[str]]:
    processed = text.lower()[:MAX_LEN]
    paths = _crf_decode_topk(emissions, transitions, start, end, 2 * k)
    result: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        words = _decode_predictions_batch([processed], [path])[0]
        key = tuple(words)
        if key in seen:
            continue
        seen.add(key)
        result.append(words)
        if len(result) == k:
            break
    return result


def _acceptable(row: dict[str, str]) -> set[str]:
    return {value.strip().lower() for value in (row["truth"], row["might_right"]) if value.strip()}


def main() -> None:
    rows = _load_rows()
    onnx_path, crf_path = _model_paths()
    with tempfile.TemporaryDirectory(prefix="dksplit-numpy-benchmark-") as temp_dir:
        converted_path = Path(temp_dir) / "dksplit-numpy.npz"
        convert_model(onnx_path, crf_path, converted_path)
        reference = Splitter(model_path=str(onnx_path), crf_path=str(crf_path), num_threads=1)
        model = FloatNumpyDKSplit(converted_path)

        top1_parity = 0
        top3_order_parity = 0
        top5_order_parity = 0
        onnx_strict_top1 = 0
        numpy_strict_top1 = 0
        onnx_lenient_top1 = 0
        numpy_lenient_top1 = 0
        onnx_lenient_top3 = 0
        numpy_lenient_top3 = 0
        onnx_lenient_top5 = 0
        numpy_lenient_top5 = 0
        onnx_only_lenient_top1 = 0
        numpy_only_lenient_top1 = 0
        max_abs_error = 0.0
        total_abs_error = 0.0
        emission_values = 0
        onnx_seconds = 0.0
        numpy_seconds = 0.0
        parity_divergences: list[tuple[str, list[str], list[str], set[str]]] = []

        for row in rows:
            text = row["prefix"].strip()
            truth = row["truth"].strip().lower()
            acceptable = _acceptable(row)

            started = time.perf_counter()
            ref_emissions = _reference_emissions(reference, text)
            onnx_seconds += time.perf_counter() - started

            started = time.perf_counter()
            np_emissions = model.emissions(text)
            numpy_seconds += time.perf_counter() - started

            difference = np.abs(ref_emissions - np_emissions)
            max_abs_error = max(max_abs_error, float(np.max(difference)))
            total_abs_error += float(np.sum(difference))
            emission_values += difference.size

            ref_top5 = _topk_words(
                text,
                ref_emissions,
                reference.transitions,
                reference.start_transitions,
                reference.end_transitions,
                5,
            )
            np_top5 = _topk_words(
                text,
                np_emissions,
                model.transitions,
                model.start_transitions,
                model.end_transitions,
                5,
            )

            ref_strings = [" ".join(words) for words in ref_top5]
            np_strings = [" ".join(words) for words in np_top5]
            ref_lenient_top1 = ref_strings[0] in acceptable
            np_lenient_top1 = np_strings[0] in acceptable

            top1_parity += ref_strings[0] == np_strings[0]
            top3_order_parity += ref_strings[:3] == np_strings[:3]
            top5_order_parity += ref_strings == np_strings
            onnx_strict_top1 += ref_strings[0] == truth
            numpy_strict_top1 += np_strings[0] == truth
            onnx_lenient_top1 += ref_lenient_top1
            numpy_lenient_top1 += np_lenient_top1
            onnx_lenient_top3 += any(candidate in acceptable for candidate in ref_strings[:3])
            numpy_lenient_top3 += any(candidate in acceptable for candidate in np_strings[:3])
            onnx_lenient_top5 += any(candidate in acceptable for candidate in ref_strings)
            numpy_lenient_top5 += any(candidate in acceptable for candidate in np_strings)
            onnx_only_lenient_top1 += ref_lenient_top1 and not np_lenient_top1
            numpy_only_lenient_top1 += np_lenient_top1 and not ref_lenient_top1

            if ref_strings[0] != np_strings[0] and len(parity_divergences) < 20:
                parity_divergences.append((text, ref_top5[0], np_top5[0], acceptable))

        count = len(rows)
        mean_abs_error = total_abs_error / emission_values
        print(f"benchmark_samples={count}")
        print(f"benchmark_top1_parity={top1_parity / count:.6%}")
        print(f"benchmark_top3_order_parity={top3_order_parity / count:.6%}")
        print(f"benchmark_top5_order_parity={top5_order_parity / count:.6%}")
        print(f"benchmark_onnx_strict_top1={onnx_strict_top1 / count:.6%}")
        print(f"benchmark_numpy_strict_top1={numpy_strict_top1 / count:.6%}")
        print(f"benchmark_onnx_lenient_top1={onnx_lenient_top1 / count:.6%}")
        print(f"benchmark_numpy_lenient_top1={numpy_lenient_top1 / count:.6%}")
        print(f"benchmark_onnx_lenient_top3={onnx_lenient_top3 / count:.6%}")
        print(f"benchmark_numpy_lenient_top3={numpy_lenient_top3 / count:.6%}")
        print(f"benchmark_onnx_lenient_top5={onnx_lenient_top5 / count:.6%}")
        print(f"benchmark_numpy_lenient_top5={numpy_lenient_top5 / count:.6%}")
        print(f"benchmark_onnx_only_lenient_top1={onnx_only_lenient_top1}")
        print(f"benchmark_numpy_only_lenient_top1={numpy_only_lenient_top1}")
        print(f"benchmark_max_abs_emission_error={max_abs_error:.9g}")
        print(f"benchmark_mean_abs_emission_error={mean_abs_error:.9g}")
        print(f"benchmark_onnx_seconds={onnx_seconds:.6f}")
        print(f"benchmark_numpy_seconds={numpy_seconds:.6f}")
        if parity_divergences:
            print("benchmark_top1_divergences:")
            for text, expected, actual, acceptable in parity_divergences:
                print(f"  {text!r}: onnx={expected!r} numpy={actual!r} acceptable={sorted(acceptable)!r}")


if __name__ == "__main__":
    main()
