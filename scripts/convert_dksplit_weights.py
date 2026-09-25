"""Convert DKSplit's published ONNX/CRF files into the NumPy runtime format.

This is a development-time utility. The resulting NPZ is consumed by
``experiments/dksplit_numpy/runtime.py`` and requires only NumPy at inference
time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dksplit
import numpy as np
import onnx
from onnx import numpy_helper

FORMAT_VERSION = 1
EXPECTED_LSTM_NODES = 3
MIN_LSTM_INPUTS = 12


def _installed_model_paths() -> tuple[Path, Path]:
    model_dir = Path(dksplit.__file__).resolve().parent / "models"
    return model_dir / "dksplit-int8.onnx", model_dir / "dksplit.npz"


def _initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def _required(initializers: dict[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return initializers[name]
    except KeyError as exc:
        message = f"required ONNX initializer is missing: {name}"
        raise ValueError(message) from exc


def convert(onnx_path: Path, crf_path: Path, output_path: Path) -> None:
    """Extract only tensors required by the fixed NumPy inference runtime."""
    model = onnx.load(onnx_path, load_external_data=False)
    initializers = _initializers(model)
    lstm_nodes = [node for node in model.graph.node if node.op_type == "DynamicQuantizeLSTM"]
    if len(lstm_nodes) != EXPECTED_LSTM_NODES:
        message = f"expected exactly {EXPECTED_LSTM_NODES} DynamicQuantizeLSTM nodes, found {len(lstm_nodes)}"
        raise ValueError(message)

    arrays: dict[str, np.ndarray] = {
        "format_version": np.asarray(FORMAT_VERSION, dtype=np.int32),
        "embedding_q": _required(initializers, "embedding.weight_quantized"),
        "embedding_scale": _required(initializers, "embedding.weight_scale"),
        "embedding_zp": _required(initializers, "embedding.weight_zero_point"),
        "projection_q": _required(initializers, "onnx::MatMul_584_quantized"),
        "projection_scale": _required(initializers, "onnx::MatMul_584_scale"),
        "projection_zp": _required(initializers, "onnx::MatMul_584_zero_point"),
        "projection_bias": _required(initializers, "hidden2tag.bias"),
    }

    for layer, node in enumerate(lstm_nodes):
        if len(node.input) < MIN_LSTM_INPUTS:
            message = f"layer {layer} has unexpected DynamicQuantizeLSTM inputs"
            raise ValueError(message)
        w_name, r_name, b_name = node.input[1:4]
        w_scale_name, w_zp_name, r_scale_name, r_zp_name = node.input[8:12]
        arrays.update(
            {
                f"l{layer}_w_q": _required(initializers, w_name),
                f"l{layer}_w_scale": _required(initializers, w_scale_name),
                f"l{layer}_w_zp": _required(initializers, w_zp_name),
                f"l{layer}_r_q": _required(initializers, r_name),
                f"l{layer}_r_scale": _required(initializers, r_scale_name),
                f"l{layer}_r_zp": _required(initializers, r_zp_name),
                f"l{layer}_bias": _required(initializers, b_name),
            }
        )

    with np.load(crf_path) as crf:
        arrays["crf_transitions"] = np.asarray(crf["transitions"], dtype=np.float32)
        arrays["crf_start_transitions"] = np.asarray(crf["start_transitions"], dtype=np.float32)
        arrays["crf_end_transitions"] = np.asarray(crf["end_transitions"], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, help="source dksplit-int8.onnx; defaults to installed DKSplit")
    parser.add_argument("--crf", type=Path, help="source dksplit.npz; defaults to installed DKSplit")
    parser.add_argument("--output", type=Path, required=True, help="destination converted NPZ")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.onnx is None and args.crf is None:
        onnx_path, crf_path = _installed_model_paths()
    elif args.onnx is not None and args.crf is not None:
        onnx_path, crf_path = args.onnx, args.crf
    else:
        message = "--onnx and --crf must be supplied together"
        raise SystemExit(message)

    convert(onnx_path, crf_path, args.output)
    print(f"source_onnx_bytes={onnx_path.stat().st_size}")
    print(f"converted_npz_bytes={args.output.stat().st_size}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
