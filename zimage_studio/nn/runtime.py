"""A small interpreter for ONNX graphs.

It walks the graph in the order the file stores it (ONNX guarantees topological
order), keeps a dictionary of live tensors and frees each one right after its
last consumer - without that, a 512x512 network exceeds the address space of a
32-bit process.

Weights are wrapped around the memory-mapped file with ``np.frombuffer``, so
they cost page cache rather than heap.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import ops
from .onnx_model import Graph, OnnxError, Tensor

ProgressCallback = Callable[[int, int, str], None]


class RuntimeError_(RuntimeError):
    """Raised when a node cannot be executed."""


def _attr(node, name, default=None):
    return node.attributes.get(name, default)


def _optional(values: Sequence[Any], index: int, default=None):
    if index < len(values) and values[index] is not None:
        return values[index]
    return default


def _as_list(value) -> List[int]:
    return [int(item) for item in np.asarray(value).ravel()]


class Interpreter:
    """Executes a :class:`~zimage_studio.nn.onnx_model.Graph`."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph
        self._weights: Dict[str, np.ndarray] = {}
        self._last_use = self._compute_last_use()

    # -- setup ---------------------------------------------------------

    def _compute_last_use(self) -> Dict[str, int]:
        last: Dict[str, int] = {}
        for index, node in enumerate(self.graph.nodes):
            for name in node.inputs:
                if name:
                    last[name] = index
        return last

    def weight(self, name: str) -> np.ndarray:
        """Materialise an initializer, caching the (zero-copy) view."""
        cached = self._weights.get(name)
        if cached is None:
            tensor = self.graph.initializers[name]
            cached = tensor.numpy()
            self._weights[name] = cached
        return cached

    # -- execution -----------------------------------------------------

    def run(
        self,
        feeds: Dict[str, np.ndarray],
        outputs: Optional[Sequence[str]] = None,
        progress: Optional[ProgressCallback] = None,
        progress_every: int = 200,
    ) -> Dict[str, np.ndarray]:
        """Run the graph and return the requested tensors."""
        wanted = list(outputs or [item.name for item in self.graph.outputs])
        values: Dict[str, np.ndarray] = {}
        for name, array in feeds.items():
            values[name] = np.asarray(array)

        missing = [name for name in self.graph.input_names if name not in values]
        if missing:
            raise RuntimeError_("missing model inputs: %s" % ", ".join(missing))

        total = len(self.graph.nodes)
        keep = set(wanted)

        for index, node in enumerate(self.graph.nodes):
            if progress is not None and (index % progress_every == 0 or index == total - 1):
                progress(index + 1, total, node.op_type)

            inputs: List[Any] = []
            for name in node.inputs:
                if not name:
                    inputs.append(None)
                elif name in values:
                    inputs.append(values[name])
                elif name in self.graph.initializers:
                    inputs.append(self.weight(name))
                else:
                    raise RuntimeError_(
                        "node %s (%s) wants unknown input %r" % (node.name, node.op_type, name)
                    )

            try:
                results = self.execute(node, inputs)
            except Exception as exc:  # noqa: BLE001 - add context, then re-raise
                raise RuntimeError_(
                    "node %d %s (%s) failed: %s" % (index, node.name or "?", node.op_type, exc)
                ) from exc

            for name, value in zip(node.outputs, results):
                if name:
                    values[name] = value

            # Free everything whose last consumer was this node.
            for name in set(node.inputs):
                if name and name not in keep and self._last_use.get(name) == index:
                    values.pop(name, None)

        result = {}
        for name in wanted:
            if name not in values:
                raise RuntimeError_("graph did not produce output %r" % name)
            result[name] = values[name]
        return result

    # -- dispatch ------------------------------------------------------

    def execute(self, node, inputs: List[Any]) -> List[Any]:
        handler = getattr(self, "_op_" + node.op_type, None)
        if handler is None:
            raise ops.OpError("operator %s is not implemented" % node.op_type)
        result = handler(node, inputs)
        return result if isinstance(result, (list, tuple)) else [result]

    # -- constants and shapes ------------------------------------------

    def _op_Constant(self, node, inputs):
        for key in ("value", "value_float", "value_int", "value_floats", "value_ints"):
            if key in node.attributes:
                value = node.attributes[key]
                if isinstance(value, Tensor):
                    return value.numpy()
                if key == "value_float":
                    return np.array(value, dtype=np.float32)
                if key == "value_int":
                    return np.array(value, dtype=np.int64)
                if key == "value_floats":
                    return np.array(value, dtype=np.float32)
                return np.array(value, dtype=np.int64)
        raise ops.OpError("Constant without a value attribute")

    def _op_ConstantOfShape(self, node, inputs):
        value = node.attributes.get("value")
        return ops.constant_of_shape(inputs[0], value.numpy() if isinstance(value, Tensor) else value)

    def _op_Shape(self, node, inputs):
        shape = np.array(inputs[0].shape, dtype=np.int64)
        start = int(_attr(node, "start", 0) or 0)
        end = _attr(node, "end")
        return shape[start : int(end) if end is not None else len(shape)]

    def _op_Size(self, node, inputs):
        return np.array(inputs[0].size, dtype=np.int64)

    def _op_Identity(self, node, inputs):
        return inputs[0]

    def _op_Dropout(self, node, inputs):
        return [inputs[0]]

    def _op_Cast(self, node, inputs):
        return ops.cast(inputs[0], to=int(_attr(node, "to", 1)))

    def _op_Reshape(self, node, inputs):
        return ops.reshape(inputs[0], inputs[1], allowzero=int(_attr(node, "allowzero", 0) or 0))

    def _op_Transpose(self, node, inputs):
        return ops.transpose(inputs[0], _attr(node, "perm"))

    def _op_Unsqueeze(self, node, inputs):
        axes = _optional(inputs, 1, _attr(node, "axes"))
        return ops.unsqueeze(inputs[0], axes)

    def _op_Squeeze(self, node, inputs):
        axes = _optional(inputs, 1, _attr(node, "axes"))
        return ops.squeeze(inputs[0], axes)

    def _op_Flatten(self, node, inputs):
        axis = int(_attr(node, "axis", 1) or 1)
        shape = inputs[0].shape
        first = int(np.prod(shape[:axis])) if axis > 0 else 1
        return inputs[0].reshape(first, -1)

    def _op_Concat(self, node, inputs):
        return np.concatenate([np.asarray(item) for item in inputs], axis=int(_attr(node, "axis", 0)))

    def _op_Split(self, node, inputs):
        axis = int(_attr(node, "axis", 0) or 0)
        split = _optional(inputs, 1, _attr(node, "split"))
        if split is None:
            count = int(_attr(node, "num_outputs", len(node.outputs)) or len(node.outputs))
            return list(np.array_split(inputs[0], count, axis=axis))
        sizes = _as_list(split)
        points = np.cumsum(sizes)[:-1]
        return list(np.split(inputs[0], points, axis=axis))

    def _op_Slice(self, node, inputs):
        if len(inputs) > 1:
            return ops.slice_op(
                inputs[0],
                _as_list(inputs[1]),
                _as_list(inputs[2]),
                _as_list(inputs[3]) if _optional(inputs, 3) is not None else None,
                _as_list(inputs[4]) if _optional(inputs, 4) is not None else None,
            )
        return ops.slice_op(
            inputs[0], _attr(node, "starts"), _attr(node, "ends"), _attr(node, "axes"), None
        )

    def _op_Gather(self, node, inputs):
        return ops.gather(inputs[0], inputs[1], axis=int(_attr(node, "axis", 0) or 0))

    def _op_GatherElements(self, node, inputs):
        return ops.gather_elements(inputs[0], inputs[1], axis=int(_attr(node, "axis", 0) or 0))

    def _op_Range(self, node, inputs):
        return ops.range_op(inputs[0], inputs[1], inputs[2])

    def _op_Expand(self, node, inputs):
        return np.broadcast_to(inputs[0], np.broadcast_shapes(inputs[0].shape, tuple(_as_list(inputs[1])))).copy()

    def _op_Tile(self, node, inputs):
        return np.tile(inputs[0], _as_list(inputs[1]))

    def _op_Where(self, node, inputs):
        return np.where(inputs[0], inputs[1], inputs[2])

    def _op_Pad(self, node, inputs):
        if len(inputs) > 1:
            value = _optional(inputs, 2, 0.0)
            axes = _optional(inputs, 3)
            return ops.pad(
                inputs[0],
                _as_list(inputs[1]),
                float(np.asarray(value).ravel()[0]) if value is not None else 0.0,
                _attr(node, "mode", b"constant"),
                _as_list(axes) if axes is not None else None,
            )
        return ops.pad(
            inputs[0], _attr(node, "pads", []), float(_attr(node, "value", 0.0) or 0.0), _attr(node, "mode", b"constant")
        )

    # -- arithmetic ----------------------------------------------------

    def _op_Add(self, node, inputs):
        return np.add(inputs[0], inputs[1])

    def _op_Sub(self, node, inputs):
        return np.subtract(inputs[0], inputs[1])

    def _op_Mul(self, node, inputs):
        return np.multiply(inputs[0], inputs[1])

    def _op_Div(self, node, inputs):
        left, right = inputs[0], inputs[1]
        if np.issubdtype(np.asarray(left).dtype, np.integer) and np.issubdtype(
            np.asarray(right).dtype, np.integer
        ):
            return (np.asarray(left) // np.asarray(right)).astype(np.asarray(left).dtype)
        return np.divide(left, right)

    def _op_Pow(self, node, inputs):
        return np.power(inputs[0], inputs[1]).astype(np.asarray(inputs[0]).dtype)

    def _op_Neg(self, node, inputs):
        return np.negative(inputs[0])

    def _op_Abs(self, node, inputs):
        return np.abs(inputs[0])

    def _op_Sqrt(self, node, inputs):
        return np.sqrt(inputs[0])

    def _op_Exp(self, node, inputs):
        return np.exp(inputs[0])

    def _op_Log(self, node, inputs):
        return np.log(inputs[0])

    def _op_Sin(self, node, inputs):
        return np.sin(inputs[0])

    def _op_Cos(self, node, inputs):
        return np.cos(inputs[0])

    def _op_Erf(self, node, inputs):
        from math import erf

        return np.vectorize(erf, otypes=[np.float32])(inputs[0])

    def _op_Reciprocal(self, node, inputs):
        return np.reciprocal(inputs[0])

    def _op_Min(self, node, inputs):
        result = inputs[0]
        for item in inputs[1:]:
            result = np.minimum(result, item)
        return result

    def _op_Max(self, node, inputs):
        result = inputs[0]
        for item in inputs[1:]:
            result = np.maximum(result, item)
        return result

    def _op_Sum(self, node, inputs):
        result = inputs[0]
        for item in inputs[1:]:
            result = result + item
        return result

    def _op_Clip(self, node, inputs):
        low = _optional(inputs, 1, _attr(node, "min"))
        high = _optional(inputs, 2, _attr(node, "max"))
        result = inputs[0]
        if low is not None:
            result = np.maximum(result, np.asarray(low, dtype=result.dtype))
        if high is not None:
            result = np.minimum(result, np.asarray(high, dtype=result.dtype))
        return result

    def _op_Equal(self, node, inputs):
        return np.equal(inputs[0], inputs[1])

    def _op_Less(self, node, inputs):
        return np.less(inputs[0], inputs[1])

    def _op_Greater(self, node, inputs):
        return np.greater(inputs[0], inputs[1])

    def _op_Not(self, node, inputs):
        return np.logical_not(inputs[0])

    def _op_And(self, node, inputs):
        return np.logical_and(inputs[0], inputs[1])

    def _op_Or(self, node, inputs):
        return np.logical_or(inputs[0], inputs[1])

    def _op_ReduceMean(self, node, inputs):
        axes = _optional(inputs, 1, _attr(node, "axes"))
        keepdims = bool(int(_attr(node, "keepdims", 1) or 0))
        axis = tuple(_as_list(axes)) if axes is not None else None
        return np.mean(inputs[0], axis=axis, keepdims=keepdims)

    def _op_ReduceSum(self, node, inputs):
        axes = _optional(inputs, 1, _attr(node, "axes"))
        keepdims = bool(int(_attr(node, "keepdims", 1) or 0))
        axis = tuple(_as_list(axes)) if axes is not None else None
        return np.sum(inputs[0], axis=axis, keepdims=keepdims)

    # -- linear algebra ------------------------------------------------

    def _op_MatMul(self, node, inputs):
        return np.matmul(inputs[0], inputs[1])

    def _op_Gemm(self, node, inputs):
        alpha = float(_attr(node, "alpha", 1.0) or 1.0)
        beta = float(_attr(node, "beta", 1.0) or 1.0)
        a = inputs[0].T if int(_attr(node, "transA", 0) or 0) else inputs[0]
        b = inputs[1].T if int(_attr(node, "transB", 0) or 0) else inputs[1]
        result = alpha * (a @ b)
        if _optional(inputs, 2) is not None:
            result = result + beta * inputs[2]
        return result

    def _op_Einsum(self, node, inputs):
        equation = _attr(node, "equation", b"")
        if isinstance(equation, (bytes, bytearray)):
            equation = equation.decode()
        return np.einsum(equation, *inputs, optimize=True)

    # -- neural network ------------------------------------------------

    def _op_Conv(self, node, inputs):
        return ops.conv(
            inputs[0],
            inputs[1],
            _optional(inputs, 2),
            strides=_attr(node, "strides"),
            pads=_attr(node, "pads"),
            dilations=_attr(node, "dilations"),
            group=int(_attr(node, "group", 1) or 1),
            auto_pad=_attr(node, "auto_pad", b"NOTSET"),
        )

    def _op_ConvTranspose(self, node, inputs):
        return ops.conv_transpose(
            inputs[0],
            inputs[1],
            _optional(inputs, 2),
            strides=_attr(node, "strides"),
            pads=_attr(node, "pads"),
            dilations=_attr(node, "dilations"),
            group=int(_attr(node, "group", 1) or 1),
            output_padding=_attr(node, "output_padding"),
            auto_pad=_attr(node, "auto_pad", b"NOTSET"),
        )

    def _op_BatchNormalization(self, node, inputs):
        return ops.batch_normalization(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            epsilon=float(_attr(node, "epsilon", 1e-5) or 1e-5),
        )

    def _op_InstanceNormalization(self, node, inputs):
        return ops.instance_normalization(
            inputs[0], inputs[1], inputs[2], epsilon=float(_attr(node, "epsilon", 1e-5) or 1e-5)
        )

    def _op_Relu(self, node, inputs):
        return ops.relu(inputs[0])

    def _op_LeakyRelu(self, node, inputs):
        return ops.leaky_relu(inputs[0], alpha=float(_attr(node, "alpha", 0.01) or 0.01))

    def _op_PRelu(self, node, inputs):
        slope = np.asarray(inputs[1], dtype=np.float32)
        if slope.ndim == 1 and inputs[0].ndim == 4:
            slope = slope.reshape(1, -1, 1, 1)
        return np.where(inputs[0] >= 0, inputs[0], inputs[0] * slope)

    def _op_Sigmoid(self, node, inputs):
        return ops.sigmoid(inputs[0])

    def _op_Tanh(self, node, inputs):
        return np.tanh(inputs[0])

    def _op_Softmax(self, node, inputs):
        return ops.softmax(inputs[0], axis=int(_attr(node, "axis", -1)))

    def _op_GlobalAveragePool(self, node, inputs):
        axes = tuple(range(2, inputs[0].ndim))
        return np.mean(inputs[0], axis=axes, keepdims=True)

    def _op_Resize(self, node, inputs):
        scales = _optional(inputs, 2)
        sizes = _optional(inputs, 3)
        return ops.resize_nearest_or_linear(inputs[0], scales, sizes, _attr(node, "mode", b"nearest"))

    def _op_Upsample(self, node, inputs):
        return ops.resize_nearest_or_linear(
            inputs[0], _optional(inputs, 1, _attr(node, "scales")), None, _attr(node, "mode", b"nearest")
        )


def load_and_run(path, feeds, progress=None):
    """Convenience wrapper used by the tests."""
    from .onnx_model import load

    return Interpreter(load(path)).run(feeds, progress=progress)
