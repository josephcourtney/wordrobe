"""Inspect the shipped DKSplit ONNX graph for the NumPy reimplementation experiment."""

from __future__ import annotations

import json
from pathlib import Path

import dksplit
import onnx
from onnx import numpy_helper


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    dims: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            dims.append(None)
    return dims


def main() -> None:
    package_dir = Path(dksplit.__file__).resolve().parent
    model_path = package_dir / "models" / "dksplit-int8.onnx"
    model = onnx.load(model_path, load_external_data=False)
    graph = model.graph

    initializers = {
        item.name: {
            "dtype": onnx.TensorProto.DataType.Name(item.data_type),
            "shape": list(item.dims),
            "nbytes": int(numpy_helper.to_array(item).nbytes),
        }
        for item in graph.initializer
    }
    nodes = []
    for index, node in enumerate(graph.node):
        attrs = {}
        for attr in node.attribute:
            value = onnx.helper.get_attribute_value(attr)
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            if hasattr(value, "tolist"):
                value = value.tolist()
            attrs[attr.name] = value
        nodes.append(
            {
                "index": index,
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
                "attributes": attrs,
            }
        )

    summary = {
        "model": str(model_path),
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "inputs": {item.name: _shape(item) for item in graph.input},
        "outputs": {item.name: _shape(item) for item in graph.output},
        "initializers": initializers,
        "nodes": nodes,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
