"""Compare NumPy DKSplit against ONNX on identifier-like synthetic inputs."""

from __future__ import annotations

import random
import tempfile
import time
from pathlib import Path

import dksplit
import numpy as np
from dksplit.split import Splitter, _crf_decode_topk, _decode_predictions_batch, _text_to_ids_fast

from convert import convert_model
from float_model import FloatNumpyDKSplit
from model import MAX_LEN

TOKENS = [
    "account",
    "admin",
    "airpods",
    "api",
    "apple",
    "archive",
    "auth",
    "battery",
    "browser",
    "cache",
    "chatgpt",
    "client",
    "config",
    "dashboard",
    "database",
    "decode",
    "device",
    "docker",
    "download",
    "driver",
    "email",
    "engine",
    "error",
    "event",
    "file",
    "github",
    "google",
    "history",
    "http",
    "image",
    "index",
    "input",
    "json",
    "launch",
    "login",
    "macos",
    "manager",
    "model",
    "monitor",
    "network",
    "openai",
    "output",
    "parser",
    "photo",
    "project",
    "python",
    "request",
    "response",
    "scroot",
    "server",
    "session",
    "settings",
    "status",
    "storage",
    "sync",
    "system",
    "token",
    "upload",
    "user",
    "version",
    "widget",
    "wordrobe",
    "xstate",
    "xml",
]

SEPARATORS = ["", "", "", "", "-", "_", "."]


def _make_inputs(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    result: list[str] = []
    for _ in range(count):
        parts = [rng.choice(TOKENS) for _ in range(rng.randint(2, 4))]
        if rng.random() < 0.25:
            parts[rng.randrange(len(parts))] += str(rng.randint(1, 999))
        text = parts[0]
        for part in parts[1:]:
            text += rng.choice(SEPARATORS) + part
        if rng.random() < 0.15:
            text = text[:1].upper() + text[1:]
        result.append(text)
    return result


def _model_paths() -> tuple[Path, Path]:
    package_dir = Path(dksplit.__file__).resolve().parent
    model_dir = package_dir / "models"
    return model_dir / "dksplit-int8.onnx", model_dir / "dksplit.npz"


def _emissions(splitter: Splitter, text: str) -> np.ndarray:
    processed = text.lower()[:MAX_LEN]
    ids = _text_to_ids_fast(processed)
    return np.asarray(splitter.session.run(None, {"chars": ids.reshape(1, -1)})[0][0], dtype=np.float32)


def _topk_words(text: str, emissions: np.ndarray, transitions: np.ndarray, start: np.ndarray, end: np.ndarray, k: int) -> list[list[str]]:
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


def main() -> None:
    texts = _make_inputs(1000, 20260924)
    onnx_path, crf_path = _model_paths()
    with tempfile.TemporaryDirectory(prefix="dksplit-numpy-realistic-") as temp_dir:
        converted_path = Path(temp_dir) / "dksplit-numpy.npz"
        convert_model(onnx_path, crf_path, converted_path)
        reference = Splitter(model_path=str(onnx_path), crf_path=str(crf_path), num_threads=1)
        model = FloatNumpyDKSplit(converted_path)

        top1_matches = 0
        top3_exact_order_matches = 0
        top3_set_matches = 0
        max_abs_error = 0.0
        total_abs_error = 0.0
        emission_values = 0
        onnx_seconds = 0.0
        numpy_seconds = 0.0
        divergences: list[tuple[str, list[list[str]], list[list[str]]]] = []

        for text in texts:
            started = time.perf_counter()
            ref_emissions = _emissions(reference, text)
            onnx_seconds += time.perf_counter() - started

            started = time.perf_counter()
            np_emissions = model.emissions(text)
            numpy_seconds += time.perf_counter() - started

            difference = np.abs(ref_emissions - np_emissions)
            max_abs_error = max(max_abs_error, float(np.max(difference)))
            total_abs_error += float(np.sum(difference))
            emission_values += difference.size

            ref_top3 = _topk_words(
                text,
                ref_emissions,
                reference.transitions,
                reference.start_transitions,
                reference.end_transitions,
                3,
            )
            np_top3 = _topk_words(
                text,
                np_emissions,
                model.transitions,
                model.start_transitions,
                model.end_transitions,
                3,
            )

            if ref_top3[0] == np_top3[0]:
                top1_matches += 1
            if ref_top3 == np_top3:
                top3_exact_order_matches += 1
            if {tuple(words) for words in ref_top3} == {tuple(words) for words in np_top3}:
                top3_set_matches += 1
            if ref_top3 != np_top3 and len(divergences) < 12:
                divergences.append((text, ref_top3, np_top3))

        count = len(texts)
        mean_abs_error = total_abs_error / emission_values
        print(f"realistic_samples={count}")
        print(f"realistic_top1_matches={top1_matches}")
        print(f"realistic_top1_parity={top1_matches / count:.6%}")
        print(f"realistic_top3_order_matches={top3_exact_order_matches}")
        print(f"realistic_top3_order_parity={top3_exact_order_matches / count:.6%}")
        print(f"realistic_top3_set_matches={top3_set_matches}")
        print(f"realistic_top3_set_parity={top3_set_matches / count:.6%}")
        print(f"realistic_max_abs_emission_error={max_abs_error:.9g}")
        print(f"realistic_mean_abs_emission_error={mean_abs_error:.9g}")
        print(f"realistic_onnx_seconds={onnx_seconds:.6f}")
        print(f"realistic_numpy_seconds={numpy_seconds:.6f}")
        if divergences:
            print("realistic_top3_divergences:")
            for text, expected, actual in divergences:
                print(f"  {text!r}: onnx={expected!r} numpy={actual!r}")


if __name__ == "__main__":
    main()
