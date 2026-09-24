"""Convert DKSplit's ONNX/CRF files into a small NumPy-only runtime archive.

The output intentionally preserves the published INT8 weights plus their
scales/zero-points.  The NumPy runtime dequantizes them once at load time.
ONNX is therefore a conversion-time dependency only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

FORMAT_VERSION = 1


def _initializers(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}


def _required(initializers: dict[str, np.ndarray], name: str) -> np.ndarray:
    try:
        return initializers[name]
    except KeyError as exc:
        raise ValueError(f"required ONNX initializer is missing: {name}") from exc


def convert_model(onnx_path: Path, crf_path: Path, output_path: Path) -> None:
    """Extract the fixed DKSplit architecture into a compact NPZ archive."""
    model = onnx.load(onnx_path, load_external_data=False)
    initializers = _initializers(model)
    lstm_nodes = [node for node in model.graph.node if node.op_type == "DynamicQuantizeLSTM"]
    if len(lstm_nodes) != 3:
        raise ValueError(f"expected exactly 3 DynamicQuantizeLSTM nodes, found {len(lstm_nodes)}")

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
        if len(node.input) < 12:
            raise ValueError(f"layer {layer} has unexpected DynamicQuantizeLSTM input list")
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
        for name in ("transitions", "start_transitions", "end_transitions"):
            arrays[f"crf_{name}"] = np.asarray(crf[name], dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="DKSplit dksplit-int8.onnx")
    parser.add_argument("--crf", type=Path, required=True, help="DKSplit dksplit.npz")
    parser.add_argument("--output", type=Path, required=True, help="destination NPZ")
    return parser


def main() -> None:
    args = _parser().parse_args()
    convert_model(args.onnx, args.crf, args.output)
    print(f"wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
