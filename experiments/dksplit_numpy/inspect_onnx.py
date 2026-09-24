"""Inspect the shipped DKSplit ONNX graph for the NumPy reimplementation experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


def _jsonable_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, onnx.TensorProto):
        array = numpy_helper.to_array(value)
        return {
            "tensor_dtype": str(array.dtype),
            "tensor_shape": list(array.shape),
            "tensor_values": array.tolist() if array.size <= 16 else None,
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_attribute(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


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
        attrs = {
            attr.name: _jsonable_attribute(onnx.helper.get_attribute_value(attr))
            for attr in node.attribute
        }
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
