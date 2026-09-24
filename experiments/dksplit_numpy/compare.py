"""Compare exploratory NumPy DKSplit backends with real ONNX Runtime."""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import dksplit
import numpy as np
from dksplit.split import Splitter, _crf_decode as reference_crf_decode
from dksplit.split import _decode_predictions_batch, _text_to_ids_fast

from convert import convert_model
from float_model import FloatNumpyDKSplit
from model import MAX_LEN, NumpyDKSplit, _crf_decode, _decode_words, text_to_ids

REPRESENTATIVE_INPUTS = [
    "chatgptlogin",
    "spotifywrapped",
    "kubernetescluster",
    "openaikey",
    "microsoftoffice",
    "pikahug",
    "thisisatest",
    "isthisacamel",
    "isthisasnorql",
    "thisisaxyzzy",
    "findacme",
    "openai",
    "scroot",
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
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" * 2,
]


class _NumpyModel(Protocol):
    transitions: np.ndarray
    start_transitions: np.ndarray
    end_transitions: np.ndarray

    def emissions(self, text: str) -> np.ndarray: ...


@dataclass
class _Metrics:
    max_abs_error: float = 0.0
    sum_abs_error: float = 0.0
    emission_values: int = 0
    segment_matches: int = 0
    representative_matches: int = 0
    random_matches: int = 0
    seconds: float = 0.0
    divergences: list[tuple[str, list[str], list[str], float]] = field(default_factory=list)

    def record(
        self,
        *,
        text: str,
        reference_words: list[str],
        actual_words: list[str],
        difference: np.ndarray,
        representative: bool,
    ) -> None:
        sample_max = float(np.max(difference)) if difference.size else 0.0
        self.max_abs_error = max(self.max_abs_error, sample_max)
        self.sum_abs_error += float(np.sum(difference))
        self.emission_values += difference.size
        if reference_words == actual_words:
            self.segment_matches += 1
            if representative:
                self.representative_matches += 1
            else:
                self.random_matches += 1
        elif len(self.divergences) < 12:
            self.divergences.append((text, reference_words, actual_words, sample_max))


def _random_inputs(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.@"
    return [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 80)))
        for _ in range(count)
    ]


def _model_paths() -> tuple[Path, Path]:
    package_dir = Path(dksplit.__file__).resolve().parent
    model_dir = package_dir / "models"
    return model_dir / "dksplit-int8.onnx", model_dir / "dksplit.npz"


def _reference_emissions(splitter: Splitter, text: str) -> np.ndarray:
    processed = text.lower()[:MAX_LEN]
    ids = _text_to_ids_fast(processed)
    return np.asarray(splitter.session.run(None, {"chars": ids.reshape(1, -1)})[0][0], dtype=np.float32)


def _reference_words(splitter: Splitter, text: str, emissions: np.ndarray) -> list[str]:
    processed = text.lower()[:MAX_LEN]
    labels = reference_crf_decode(
        emissions[None, :, :],
        splitter.transitions,
        splitter.start_transitions,
        splitter.end_transitions,
    )
    return _decode_predictions_batch([processed], labels)[0]


def _numpy_words(model: _NumpyModel, text: str, emissions: np.ndarray) -> list[str]:
    processed = text.lower()[:MAX_LEN]
    labels = _crf_decode(emissions, model.transitions, model.start_transitions, model.end_transitions)
    return _decode_words(processed, labels)


def _print_metrics(
    name: str,
    metrics: _Metrics,
    *,
    total: int,
    representative_count: int,
    random_count: int,
) -> None:
    mean_abs_error = metrics.sum_abs_error / metrics.emission_values if metrics.emission_values else 0.0
    print(f"{name}_max_abs_emission_error={metrics.max_abs_error:.9g}")
    print(f"{name}_mean_abs_emission_error={mean_abs_error:.9g}")
    print(f"{name}_segmentation_matches={metrics.segment_matches}")
    print(f"{name}_segmentation_parity={metrics.segment_matches / total:.6%}")
    print(f"{name}_representative_matches={metrics.representative_matches}")
    print(f"{name}_representative_parity={metrics.representative_matches / representative_count:.6%}")
    if random_count:
        print(f"{name}_random_matches={metrics.random_matches}")
        print(f"{name}_random_parity={metrics.random_matches / random_count:.6%}")
    print(f"{name}_seconds={metrics.seconds:.6f}")
    if metrics.divergences:
        print(f"{name}_divergences:")
        for text, expected, actual, error in metrics.divergences:
            print(f"  {text!r}: onnx={expected!r} numpy={actual!r} max_abs_error={error:.9g}")


def compare(random_count: int, seed: int, *, include_quantized: bool) -> int:
    onnx_path, crf_path = _model_paths()
    with tempfile.TemporaryDirectory(prefix="dksplit-numpy-") as temp_dir:
        converted_path = Path(temp_dir) / "dksplit-numpy.npz"
        convert_model(onnx_path, crf_path, converted_path)
        print(f"onnx_bytes={onnx_path.stat().st_size}")
        print(f"converted_bytes={converted_path.stat().st_size}")

        reference = Splitter(model_path=str(onnx_path), crf_path=str(crf_path), num_threads=1)
        models: list[tuple[str, _NumpyModel]] = [("float32", FloatNumpyDKSplit(converted_path))]
        if include_quantized:
            models.append(("quantized", NumpyDKSplit(converted_path)))
        metrics = {name: _Metrics() for name, _model in models}

        representative_count = len(REPRESENTATIVE_INPUTS)
        texts = REPRESENTATIVE_INPUTS + _random_inputs(random_count, seed)
        onnx_seconds = 0.0

        for index, text in enumerate(texts):
            expected_ids = _text_to_ids_fast(text.lower()[:MAX_LEN])
            actual_ids = text_to_ids(text)
            if not np.array_equal(expected_ids, actual_ids):
                raise AssertionError(f"character mapping mismatch for {text!r}")

            started = time.perf_counter()
            ref_emissions = _reference_emissions(reference, text)
            onnx_seconds += time.perf_counter() - started
            reference_words = _reference_words(reference, text, ref_emissions)

            for name, model in models:
                started = time.perf_counter()
                np_emissions = model.emissions(text)
                metrics[name].seconds += time.perf_counter() - started
                if ref_emissions.shape != np_emissions.shape:
                    raise AssertionError(
                        f"{name} emission shape mismatch for {text!r}: "
                        f"{ref_emissions.shape} != {np_emissions.shape}"
                    )
                difference = np.abs(ref_emissions - np_emissions)
                actual_words = _numpy_words(model, text, np_emissions)
                metrics[name].record(
                    text=text,
                    reference_words=reference_words,
                    actual_words=actual_words,
                    difference=difference,
                    representative=index < representative_count,
                )

        total = len(texts)
        print(f"samples={total}")
        print(f"representative_samples={representative_count}")
        print(f"random_samples={random_count}")
        print(f"onnx_seconds={onnx_seconds:.6f}")
        for name, _model in models:
            _print_metrics(
                name,
                metrics[name],
                total=total,
                representative_count=representative_count,
                random_count=random_count,
            )
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument(
        "--skip-quantized",
        action="store_true",
        help="run only the practical float32 backend",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    raise SystemExit(compare(args.random_count, args.seed, include_quantized=not args.skip_quantized))


if __name__ == "__main__":
    main()
