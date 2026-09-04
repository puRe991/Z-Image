"""Read an ONNX model without the ``onnx`` package.

Only the parts an inference engine needs are decoded: the graph's nodes,
their attributes, and the initializers (weights).  Weight bytes stay as
memoryview slices into the memory-mapped file until a node asks for them, which
keeps the resident set small on a 32-bit machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import mmap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .protobuf import (
    ProtobufError,
    as_signed,
    as_text,
    first,
    parse_message,
    unpack_varints,
)

# -- ONNX field numbers ------------------------------------------------------

_MODEL_GRAPH = 7
_MODEL_OPSET = 8
_MODEL_IR = 1

_GRAPH_NODE = 1
_GRAPH_NAME = 2
_GRAPH_INITIALIZER = 5
_GRAPH_INPUT = 11
_GRAPH_OUTPUT = 12

_NODE_INPUT = 1
_NODE_OUTPUT = 2
_NODE_NAME = 3
_NODE_OP_TYPE = 4
_NODE_ATTRIBUTE = 5

_ATTR_NAME = 1
_ATTR_F = 2
_ATTR_I = 3
_ATTR_S = 4
_ATTR_T = 5
_ATTR_FLOATS = 7
_ATTR_INTS = 8
_ATTR_STRINGS = 9
_ATTR_TYPE = 20

_TENSOR_DIMS = 1
_TENSOR_DATA_TYPE = 2
_TENSOR_FLOAT_DATA = 4
_TENSOR_INT32_DATA = 5
_TENSOR_INT64_DATA = 7
_TENSOR_NAME = 8
_TENSOR_RAW_DATA = 9
_TENSOR_DOUBLE_DATA = 10

_VALUE_NAME = 1
_VALUE_TYPE = 2
_TYPE_TENSOR = 1
_TENSOR_TYPE_ELEM = 1
_TENSOR_TYPE_SHAPE = 2
_SHAPE_DIM = 1
_DIM_VALUE = 1
_DIM_PARAM = 2

# ONNX TensorProto.DataType -> (numpy dtype string, bytes per element)
DATA_TYPES = {
    1: ("float32", 4),
    2: ("uint8", 1),
    3: ("int8", 1),
    4: ("uint16", 2),
    5: ("int16", 2),
    6: ("int32", 4),
    7: ("int64", 8),
    9: ("bool", 1),
    10: ("float16", 2),
    11: ("float64", 8),
    12: ("uint32", 4),
    13: ("uint64", 8),
}

_ATTRIBUTE_KINDS = {1: "f", 2: "i", 3: "s", 4: "t", 6: "floats", 7: "ints", 8: "strings"}


class OnnxError(ValueError):
    """Raised for models this reader cannot handle."""


@dataclass
class Tensor:
    """An initializer: shape, dtype and the raw bytes, still unparsed."""

    name: str
    dims: Tuple[int, ...]
    data_type: int
    raw: Optional[memoryview] = None
    values: Optional[List[Any]] = None

    @property
    def dtype(self) -> str:
        try:
            return DATA_TYPES[self.data_type][0]
        except KeyError:
            raise OnnxError("unsupported tensor data type %d" % self.data_type) from None

    def numpy(self):
        """Materialise the tensor as a numpy array (weights stay read-only)."""
        import numpy as np

        dtype, _size = DATA_TYPES.get(self.data_type, (None, None))
        if dtype is None:
            raise OnnxError("unsupported tensor data type %d" % self.data_type)
        if self.raw is not None:
            array = np.frombuffer(self.raw, dtype=dtype)
        else:
            array = np.array(self.values or [], dtype=dtype)
        expected = 1
        for dim in self.dims:
            expected *= dim
        if array.size != expected:
            raise OnnxError(
                "tensor %s: expected %d elements, found %d" % (self.name, expected, array.size)
            )
        return array.reshape(self.dims) if self.dims else array.reshape(())


@dataclass
class Node:
    op_type: str
    name: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)

    def attr(self, key: str, default=None):
        return self.attributes.get(key, default)


@dataclass
class ValueInfo:
    name: str
    elem_type: int = 0
    shape: Tuple[Any, ...] = ()


@dataclass
class Graph:
    nodes: List[Node]
    initializers: Dict[str, Tensor]
    inputs: List[ValueInfo]
    outputs: List[ValueInfo]
    name: str = ""
    opset: int = 0
    ir_version: int = 0

    @property
    def input_names(self) -> List[str]:
        """Graph inputs that are not already provided as initializers."""
        return [item.name for item in self.inputs if item.name not in self.initializers]


def _parse_tensor(data: memoryview) -> Tensor:
    fields = parse_message(data)
    dims: List[int] = []
    for value in fields.get(_TENSOR_DIMS, []):
        if isinstance(value, int):
            dims.append(value)
        else:  # packed
            dims.extend(unpack_varints(value))
    data_type = first(fields, _TENSOR_DATA_TYPE, 1)
    name = as_text(first(fields, _TENSOR_NAME, b""))
    raw = first(fields, _TENSOR_RAW_DATA)

    values: Optional[List[Any]] = None
    if raw is None:
        import struct

        if _TENSOR_FLOAT_DATA in fields:
            values = []
            for chunk in fields[_TENSOR_FLOAT_DATA]:
                if isinstance(chunk, int):
                    values.append(struct.unpack("<f", chunk.to_bytes(4, "little"))[0])
                else:
                    values.extend(
                        struct.unpack("<%df" % (len(chunk) // 4), bytes(chunk))
                    )
        elif _TENSOR_INT64_DATA in fields:
            values = []
            for chunk in fields[_TENSOR_INT64_DATA]:
                if isinstance(chunk, int):
                    values.append(as_signed(chunk))
                else:
                    values.extend(as_signed(item) for item in unpack_varints(chunk))
        elif _TENSOR_INT32_DATA in fields:
            values = []
            for chunk in fields[_TENSOR_INT32_DATA]:
                if isinstance(chunk, int):
                    values.append(as_signed(chunk, 32))
                else:
                    values.extend(as_signed(item, 32) for item in unpack_varints(chunk))
        elif _TENSOR_DOUBLE_DATA in fields:
            values = []
            for chunk in fields[_TENSOR_DOUBLE_DATA]:
                if isinstance(chunk, int):
                    values.append(struct.unpack("<d", chunk.to_bytes(8, "little"))[0])
                else:
                    values.extend(struct.unpack("<%dd" % (len(chunk) // 8), bytes(chunk)))
        else:
            values = []

    return Tensor(name=name, dims=tuple(dims), data_type=data_type, raw=raw, values=values)


def _parse_attribute(data: memoryview) -> Tuple[str, Any]:
    fields = parse_message(data)
    name = as_text(first(fields, _ATTR_NAME, b""))
    kind = _ATTRIBUTE_KINDS.get(first(fields, _ATTR_TYPE, 0))

    if kind == "f" or (kind is None and _ATTR_F in fields):
        import struct

        raw = first(fields, _ATTR_F, 0)
        return name, struct.unpack("<f", int(raw).to_bytes(4, "little"))[0]
    if kind == "i" or (kind is None and _ATTR_I in fields):
        return name, as_signed(int(first(fields, _ATTR_I, 0)))
    if kind == "s" or (kind is None and _ATTR_S in fields):
        return name, bytes(first(fields, _ATTR_S, b""))
    if kind == "t" or (kind is None and _ATTR_T in fields):
        return name, _parse_tensor(first(fields, _ATTR_T))
    if kind == "ints" or (kind is None and _ATTR_INTS in fields):
        values: List[int] = []
        for chunk in fields.get(_ATTR_INTS, []):
            if isinstance(chunk, int):
                values.append(as_signed(chunk))
            else:
                values.extend(as_signed(item) for item in unpack_varints(chunk))
        return name, values
    if kind == "floats" or (kind is None and _ATTR_FLOATS in fields):
        import struct

        values_f: List[float] = []
        for chunk in fields.get(_ATTR_FLOATS, []):
            if isinstance(chunk, int):
                values_f.append(struct.unpack("<f", chunk.to_bytes(4, "little"))[0])
            else:
                values_f.extend(struct.unpack("<%df" % (len(chunk) // 4), bytes(chunk)))
        return name, values_f
    if kind == "strings":
        return name, [bytes(item) for item in fields.get(_ATTR_STRINGS, [])]
    return name, None


def _parse_node(data: memoryview) -> Node:
    fields = parse_message(data)
    attributes: Dict[str, Any] = {}
    for raw_attribute in fields.get(_NODE_ATTRIBUTE, []):
        key, value = _parse_attribute(raw_attribute)
        attributes[key] = value
    return Node(
        op_type=as_text(first(fields, _NODE_OP_TYPE, b"")),
        name=as_text(first(fields, _NODE_NAME, b"")),
        inputs=[as_text(item) for item in fields.get(_NODE_INPUT, [])],
        outputs=[as_text(item) for item in fields.get(_NODE_OUTPUT, [])],
        attributes=attributes,
    )


def _parse_value_info(data: memoryview) -> ValueInfo:
    fields = parse_message(data)
    name = as_text(first(fields, _VALUE_NAME, b""))
    info = ValueInfo(name=name)
    type_field = first(fields, _VALUE_TYPE)
    if type_field is None:
        return info
    tensor_type = first(parse_message(type_field), _TYPE_TENSOR)
    if tensor_type is None:
        return info
    tensor_fields = parse_message(tensor_type)
    info.elem_type = int(first(tensor_fields, _TENSOR_TYPE_ELEM, 0) or 0)
    shape_field = first(tensor_fields, _TENSOR_TYPE_SHAPE)
    if shape_field is None:
        return info
    dims = []
    for raw_dim in parse_message(shape_field).get(_SHAPE_DIM, []):
        dim_fields = parse_message(raw_dim)
        if _DIM_VALUE in dim_fields:
            dims.append(int(first(dim_fields, _DIM_VALUE)))
        else:
            dims.append(as_text(first(dim_fields, _DIM_PARAM, b"?")))
    info.shape = tuple(dims)
    return info


def load(path) -> Graph:
    """Memory-map an ``.onnx`` file and decode its graph."""
    file_path = Path(path)
    if not file_path.is_file():
        raise OnnxError("model file not found: %s" % file_path)

    handle = open(file_path, "rb")
    try:
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ)
    except (OSError, ValueError):  # tiny files / platforms without mmap
        mapped = handle.read()
    view = memoryview(mapped)

    try:
        model_fields = parse_message(view)
    except ProtobufError as exc:
        raise OnnxError("not a valid ONNX file: %s" % exc) from exc

    graph_field = first(model_fields, _MODEL_GRAPH)
    if graph_field is None:
        raise OnnxError("ONNX model without a graph")

    opset = 0
    for raw_opset in model_fields.get(_MODEL_OPSET, []):
        opset_fields = parse_message(raw_opset)
        domain = as_text(first(opset_fields, 1, b""))
        if not domain:  # the default ONNX domain
            opset = int(first(opset_fields, 2, 0) or 0)

    graph_fields = parse_message(graph_field)
    nodes = [_parse_node(item) for item in graph_fields.get(_GRAPH_NODE, [])]
    initializers = {}
    for item in graph_fields.get(_GRAPH_INITIALIZER, []):
        tensor = _parse_tensor(item)
        initializers[tensor.name] = tensor

    graph = Graph(
        nodes=nodes,
        initializers=initializers,
        inputs=[_parse_value_info(item) for item in graph_fields.get(_GRAPH_INPUT, [])],
        outputs=[_parse_value_info(item) for item in graph_fields.get(_GRAPH_OUTPUT, [])],
        name=as_text(first(graph_fields, _GRAPH_NAME, b"")),
        opset=opset,
        ir_version=int(first(model_fields, _MODEL_IR, 0) or 0),
    )
    # Keep the mapping alive for as long as the graph references its bytes.
    graph._mmap = mapped  # type: ignore[attr-defined]
    graph._handle = handle  # type: ignore[attr-defined]
    return graph
