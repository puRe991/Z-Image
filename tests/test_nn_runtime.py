"""The ONNX reader and the NumPy interpreter.

Two layers of checking:

* a committed 4 kB model with a reference output, so the core path is verified
  wherever NumPy exists - no 200 MB download, no onnxruntime,
* per-operator comparisons against onnxruntime when it happens to be installed,
  which is what caught the negative-step Slice bug.
"""

from __future__ import annotations

import pathlib

import pytest

np = pytest.importorskip("numpy")

from zimage_studio.nn import onnx_model  # noqa: E402
from zimage_studio.nn.ops import OpError, conv, conv_transpose, pad, slice_op  # noqa: E402
from zimage_studio.nn.protobuf import ProtobufError, parse_message, read_varint  # noqa: E402
from zimage_studio.nn.runtime import Interpreter, RuntimeError_  # noqa: E402

DATA = pathlib.Path(__file__).parent / "data"
TINY_MODEL = DATA / "tiny_model.onnx"


# ----------------------------------------------------------------------
# protobuf
# ----------------------------------------------------------------------


def test_varint_roundtrip():
    assert read_varint(memoryview(b"\x00"), 0) == (0, 1)
    assert read_varint(memoryview(b"\x01"), 0) == (1, 1)
    assert read_varint(memoryview(b"\xac\x02"), 0) == (300, 2)


def test_truncated_varint_is_reported():
    with pytest.raises(ProtobufError):
        read_varint(memoryview(b"\x80"), 0)


def test_unknown_wire_type_is_reported():
    with pytest.raises(ProtobufError):
        parse_message(memoryview(b"\x07"))  # field 0, wire type 7


# ----------------------------------------------------------------------
# reading a model
# ----------------------------------------------------------------------


def test_reads_the_graph_structure():
    graph = onnx_model.load(TINY_MODEL)
    assert graph.opset == 17
    assert [node.op_type for node in graph.nodes][:3] == ["Conv", "BatchNormalization", "Relu"]
    assert graph.input_names == ["x"]
    assert [item.name for item in graph.outputs] == ["y", "rev"]


def test_reads_weights_and_attributes():
    graph = onnx_model.load(TINY_MODEL)
    weight = graph.initializers["w1"].numpy()
    assert weight.shape == (8, 3, 3, 3)
    assert weight.dtype == np.float32
    conv_node = graph.nodes[0]
    assert conv_node.attr("kernel_shape") == [3, 3]
    assert conv_node.attr("pads") == [1, 1, 1, 1]
    assert graph.nodes[3].attr("mode") == b"reflect"


def test_missing_file_is_reported():
    with pytest.raises(onnx_model.OnnxError):
        onnx_model.load(DATA / "does-not-exist.onnx")


def test_garbage_file_is_reported(tmp_path):
    broken = tmp_path / "broken.onnx"
    broken.write_bytes(b"\xff" * 64)
    with pytest.raises(onnx_model.OnnxError):
        onnx_model.load(broken)


# ----------------------------------------------------------------------
# running a model
# ----------------------------------------------------------------------


def test_matches_the_reference_output():
    """The committed reference was produced by onnxruntime."""
    graph = onnx_model.load(TINY_MODEL)
    x = np.load(DATA / "tiny_input.npy")
    expected = np.load(DATA / "tiny_output.npy")
    result = Interpreter(graph).run({"x": x})
    assert result["y"].shape == expected.shape
    assert np.abs(result["y"] - expected).max() < 2e-5
    # the reversed-slice branch: the input shape backwards
    assert list(result["rev"]) == [16, 16, 8, 1]


def test_progress_is_reported_and_can_abort():
    graph = onnx_model.load(TINY_MODEL)
    x = np.load(DATA / "tiny_input.npy")
    seen = []

    Interpreter(graph).run({"x": x}, progress=lambda done, total, op: seen.append((done, total)))
    assert seen and seen[-1][0] == seen[-1][1] == len(graph.nodes)

    class _Stop(Exception):
        pass

    def stop(done, total, op):
        raise _Stop()

    with pytest.raises(_Stop):
        Interpreter(graph).run({"x": x}, progress=stop)


def test_missing_input_is_reported():
    graph = onnx_model.load(TINY_MODEL)
    with pytest.raises(RuntimeError_, match="missing model inputs"):
        Interpreter(graph).run({})


def test_tensors_are_freed_after_their_last_use():
    """Without this the network does not fit in a 32-bit address space."""
    graph = onnx_model.load(TINY_MODEL)
    interpreter = Interpreter(graph)
    assert interpreter._last_use["c1"] < len(graph.nodes) - 1
    x = np.load(DATA / "tiny_input.npy")
    result = interpreter.run({"x": x}, outputs=["y"])
    assert set(result) == {"y"}


def test_unknown_operator_is_reported():
    graph = onnx_model.load(TINY_MODEL)
    graph.nodes[0].op_type = "Frobnicate"
    with pytest.raises(RuntimeError_, match="Frobnicate"):
        Interpreter(graph).run({"x": np.load(DATA / "tiny_input.npy")})


# ----------------------------------------------------------------------
# individual operators
# ----------------------------------------------------------------------


def test_slice_with_negative_step_reaches_the_first_element():
    """ONNX's "past the beginning" is not Python's negative index."""
    values = np.arange(5)
    assert list(slice_op(values, [-1], [-(2**63) + 1], [0], [-1])) == [4, 3, 2, 1, 0]
    assert list(slice_op(values, [3], [0], [0], [-1])) == [3, 2, 1]


def test_slice_clamps_out_of_range_bounds():
    values = np.arange(4)
    assert list(slice_op(values, [0], [2**62], [0], [1])) == [0, 1, 2, 3]
    assert list(slice_op(values, [-99], [99], [0], [1])) == [0, 1, 2, 3]


def test_empty_pads_is_a_no_op():
    values = np.ones((1, 1, 4, 4), dtype=np.float32)
    assert pad(values, []).shape == values.shape


def test_pad_modes():
    values = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)
    assert pad(values, [0, 0, 1, 1, 0, 0, 1, 1], 5.0).shape == (1, 1, 4, 4)
    assert pad(values, [0, 0, 1, 1, 0, 0, 1, 1], 0.0, b"reflect")[0, 0, 0, 0] == 3
    with pytest.raises(OpError):
        pad(values, [0, 0, 1, 1, 0, 0, 1, 1], 0.0, b"nonsense")


def _reference_conv(x, w, b=None, stride=1, padding=0, dilation=1, groups=1):
    """A slow, obviously-correct convolution to compare against."""
    n, c, h, width = x.shape
    out_channels, in_per_group, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    out_h = (xp.shape[2] - ((kh - 1) * dilation + 1)) // stride + 1
    out_w = (xp.shape[3] - ((kw - 1) * dilation + 1)) // stride + 1
    out = np.zeros((n, out_channels, out_h, out_w), dtype=np.float32)
    out_per_group = out_channels // groups
    for image in range(n):
        for channel in range(out_channels):
            group = channel // out_per_group
            for y in range(out_h):
                for x_index in range(out_w):
                    total = 0.0
                    for ci in range(in_per_group):
                        for ky in range(kh):
                            for kx in range(kw):
                                total += (
                                    xp[image, group * in_per_group + ci, y * stride + ky * dilation,
                                       x_index * stride + kx * dilation]
                                    * w[channel, ci, ky, kx]
                                )
                    out[image, channel, y, x_index] = total + (b[channel] if b is not None else 0.0)
    return out


@pytest.mark.parametrize(
    "stride,padding,dilation,groups",
    [(1, 0, 1, 1), (1, 1, 1, 1), (2, 1, 1, 1), (1, 2, 2, 1), (1, 1, 1, 2)],
)
def test_conv_matches_a_naive_implementation(stride, padding, dilation, groups):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((1, 4, 9, 9)).astype(np.float32)
    w = rng.standard_normal((6, 4 // groups, 3, 3)).astype(np.float32)
    b = rng.standard_normal(6).astype(np.float32)
    mine = conv(
        x, w, b,
        strides=[stride, stride],
        pads=[padding, padding, padding, padding],
        dilations=[dilation, dilation],
        group=groups,
    )
    expected = _reference_conv(x, w, b, stride, padding, dilation, groups)
    assert mine.shape == expected.shape
    assert np.abs(mine - expected).max() < 2e-4


def test_conv_blocking_does_not_change_the_result():
    """The row-block loop is a memory optimisation, not a numerical one."""
    from zimage_studio.nn import ops

    rng = np.random.default_rng(5)
    x = rng.standard_normal((1, 3, 24, 24)).astype(np.float32)
    w = rng.standard_normal((5, 3, 3, 3)).astype(np.float32)
    full = conv(x, w, pads=[1, 1, 1, 1])
    original = ops.CONV_BLOCK_BYTES
    try:
        ops.CONV_BLOCK_BYTES = 1  # forces one output row per block
        blocked = conv(x, w, pads=[1, 1, 1, 1])
    finally:
        ops.CONV_BLOCK_BYTES = original
    assert np.abs(full - blocked).max() == 0


def test_conv_transpose_upsamples():
    rng = np.random.default_rng(6)
    x = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    w = rng.standard_normal((2, 3, 2, 2)).astype(np.float32)
    out = conv_transpose(x, w, strides=[2, 2])
    assert out.shape == (1, 3, 8, 8)


# ----------------------------------------------------------------------
# cross-check against onnxruntime, when it is available
# ----------------------------------------------------------------------


def _run_both(model_bytes, feeds, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    reference = session.run(None, feeds)[0]
    path = tmp_path / "case.onnx"
    path.write_bytes(model_bytes)
    mine = Interpreter(onnx_model.load(path)).run(feeds)
    return mine[list(mine)[0]], reference


@pytest.mark.parametrize(
    "op,attrs,shapes",
    [
        ("Relu", {}, [(1, 3, 5, 5)]),
        ("Sigmoid", {}, [(1, 3, 5, 5)]),
        ("Tanh", {}, [(2, 4)]),
        ("Softmax", {"axis": 1}, [(2, 5)]),
        ("Transpose", {"perm": [0, 3, 1, 2]}, [(1, 4, 5, 3)]),
        ("Sqrt", {}, [(3, 3)]),
        ("Neg", {}, [(3, 3)]),
        ("Exp", {}, [(3, 3)]),
    ],
)
def test_unary_operators_match_onnxruntime(op, attrs, shapes, tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto

    rng = np.random.default_rng(9)
    inputs = [np.abs(rng.standard_normal(shape)).astype(np.float32) for shape in shapes]
    node = helper.make_node(op, ["a"], ["out"], **attrs)
    graph = helper.make_graph(
        [node],
        "case",
        [helper.make_tensor_value_info("a", TensorProto.FLOAT, list(shapes[0]))],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 8
    mine, reference = _run_both(model.SerializeToString(), {"a": inputs[0]}, tmp_path)
    assert np.abs(mine - reference).max() < 1e-5


def test_einsum_matches_onnxruntime(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import helper, TensorProto

    rng = np.random.default_rng(11)
    a = rng.standard_normal((2, 3, 4)).astype(np.float32)
    b = rng.standard_normal((2, 4, 5)).astype(np.float32)
    node = helper.make_node("Einsum", ["a", "b"], ["out"], equation="bij,bjk->bik")
    graph = helper.make_graph(
        [node],
        "case",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 4, 5]),
        ],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 8
    mine, reference = _run_both(model.SerializeToString(), {"a": a, "b": b}, tmp_path)
    assert np.abs(mine - reference).max() < 1e-5


# ----------------------------------------------------------------------
# pooling
# ----------------------------------------------------------------------


def test_max_pool_matches_a_naive_implementation():
    from zimage_studio.nn.ops import max_pool

    rng = np.random.default_rng(12)
    x = rng.standard_normal((1, 2, 6, 6)).astype(np.float32)
    result = max_pool(x, kernel_shape=[2, 2], strides=[2, 2])
    assert result.shape == (1, 2, 3, 3)
    for channel in range(2):
        for y in range(3):
            for x_index in range(3):
                window = x[0, channel, y * 2 : y * 2 + 2, x_index * 2 : x_index * 2 + 2]
                assert abs(result[0, channel, y, x_index] - window.max()) < 1e-6


def test_average_pool_matches_a_naive_implementation():
    from zimage_studio.nn.ops import average_pool

    rng = np.random.default_rng(13)
    x = rng.standard_normal((1, 1, 4, 4)).astype(np.float32)
    result = average_pool(x, kernel_shape=[2, 2], strides=[2, 2])
    assert abs(result[0, 0, 0, 0] - x[0, 0, :2, :2].mean()) < 1e-6


def test_max_pool_with_padding_and_ceil_mode():
    from zimage_studio.nn.ops import max_pool

    x = np.arange(25, dtype=np.float32).reshape(1, 1, 5, 5)
    padded = max_pool(x, kernel_shape=[3, 3], strides=[2, 2], pads=[1, 1, 1, 1])
    assert padded.shape == (1, 1, 3, 3)
    ceiled = max_pool(x, kernel_shape=[2, 2], strides=[2, 2], ceil_mode=1)
    assert ceiled.shape == (1, 1, 3, 3)
