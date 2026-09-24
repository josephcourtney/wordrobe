"""Compare the exploratory NumPy DKSplit implementation with real ONNX Runtime."""

from __future__ import annotations

import argparse
import random
import tempfile
import time
from pathlib import Path

import dksplit
import numpy as np
from dksplit.split import Splitter, _text_to_ids_fast

from convert import convert_model
from model import MAX_LEN, NumpyDKSplit, text_to_ids

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
]


def _random_inputs(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 32)))
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


def compare(random_count: int, seed: int) -> int:
    onnx_path, crf_path = _model_paths()
    with tempfile.TemporaryDirectory(prefix="dksplit-numpy-") as temp_dir:
        converted_path = Path(temp_dir) / "dksplit-numpy.npz"
        convert_model(onnx_path, crf_path, converted_path)
        print(f"onnx_bytes={onnx_path.stat().st_size}")
        print(f"converted_bytes={converted_path.stat().st_size}")

        numpy_model = NumpyDKSplit(converted_path)
        reference = Splitter(model_path=str(onnx_path), crf_path=str(crf_path), num_threads=1)

        texts = REPRESENTATIVE_INPUTS + _random_inputs(random_count, seed)
        max_abs_error = 0.0
        sum_abs_error = 0.0
        emission_values = 0
        segment_matches = 0
        divergences: list[tuple[str, list[str], list[str], float]] = []
        numpy_seconds = 0.0
        onnx_seconds = 0.0

        for text in texts:
            expected_ids = _text_to_ids_fast(text.lower()[:MAX_LEN])
            actual_ids = text_to_ids(text)
            if not np.array_equal(expected_ids, actual_ids):
                raise AssertionError(f"character mapping mismatch for {text!r}")

            started = time.perf_counter()
            ref_emissions = _reference_emissions(reference, text)
            onnx_seconds += time.perf_counter() - started

            started = time.perf_counter()
            np_emissions = numpy_model.emissions(text)
            numpy_seconds += time.perf_counter() - started

            if ref_emissions.shape != np_emissions.shape:
                raise AssertionError(
                    f"emission shape mismatch for {text!r}: {ref_emissions.shape} != {np_emissions.shape}"
                )
            difference = np.abs(ref_emissions - np_emissions)
            sample_max = float(np.max(difference)) if difference.size else 0.0
            max_abs_error = max(max_abs_error, sample_max)
            sum_abs_error += float(np.sum(difference))
            emission_values += difference.size

            reference_words = reference.split(text)
            numpy_words = numpy_model.split(text)
            if reference_words == numpy_words:
                segment_matches += 1
            elif len(divergences) < 12:
                divergences.append((text, reference_words, numpy_words, sample_max))

        total = len(texts)
        mean_abs_error = sum_abs_error / emission_values if emission_values else 0.0
        print(f"samples={total}")
        print(f"emission_values={emission_values}")
        print(f"max_abs_emission_error={max_abs_error:.9g}")
        print(f"mean_abs_emission_error={mean_abs_error:.9g}")
        print(f"segmentation_matches={segment_matches}")
        print(f"segmentation_parity={segment_matches / total:.6%}")
        print(f"onnx_seconds={onnx_seconds:.6f}")
        print(f"numpy_seconds={numpy_seconds:.6f}")
        if divergences:
            print("divergences:")
            for text, expected, actual, error in divergences:
                print(f"  {text!r}: onnx={expected!r} numpy={actual!r} max_abs_error={error:.9g}")
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260924)
    return parser


def main() -> None:
    args = _parser().parse_args()
    raise SystemExit(compare(args.random_count, args.seed))


if __name__ == "__main__":
    main()
