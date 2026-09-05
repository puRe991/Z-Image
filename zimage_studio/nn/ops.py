"""ONNX operator implementations on top of NumPy.

The set is deliberately small: exactly the operators the inpainting network
uses.  Two things matter more than generality here:

* **Memory.**  A 32-bit process has about 2 GB of address space, so the
  convolution builds its patch matrix in row blocks instead of materialising
  one 600 MB array.
* **Fidelity.**  Every operator follows the ONNX specification closely enough
  that the whole model reproduces onnxruntime's output; the test-suite compares
  against it node by node.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Roughly how much memory one convolution block may use for its patch matrix.
CONV_BLOCK_BYTES = 48 * 1024 * 1024


class OpError(NotImplementedError):
    """Raised for an operator or attribute combination that is not supported."""


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _pairs(values: Optional[Sequence[int]], rank: int, default: int = 0) -> List[int]:
    if not values:
        return [default] * rank
    return list(values)


def _auto_pad_sizes(auto_pad: bytes, input_size: int, kernel: int, stride: int, dilation: int):
    """Padding for SAME_UPPER / SAME_LOWER, split over both sides."""
    output_size = int(np.ceil(input_size / stride))
    effective = (kernel - 1) * dilation + 1
    total = max(0, (output_size - 1) * stride + effective - input_size)
    if auto_pad == b"SAME_UPPER":
        return total // 2, total - total // 2
    return total - total // 2, total // 2


def _spatial_pad(x: np.ndarray, pads: Sequence[int]) -> np.ndarray:
    """Zero-pad the two trailing (spatial) axes."""
    top, left, bottom, right = pads
    if not any(pads):
        return x
    if min(pads) < 0:  # negative padding crops
        top_c, left_c = max(0, -top), max(0, -left)
        bottom_c, right_c = max(0, -bottom), max(0, -right)
        x = x[:, :, top_c : x.shape[2] - bottom_c, left_c : x.shape[3] - right_c]
        top, left, bottom, right = max(0, top), max(0, left), max(0, bottom), max(0, right)
        if not any((top, left, bottom, right)):
            return x
    return np.pad(x, ((0, 0), (0, 0), (top, bottom), (left, right)))


def _patches(x: np.ndarray, kernel, stride, dilation, out_h: int, out_w: int) -> np.ndarray:
    """A strided view of every convolution patch - no data is copied here."""
    kh, kw = kernel
    sh, sw = stride
    dh, dw = dilation
    n, c, _h, _w = x.shape
    sn, sc, sy, sx = x.strides
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(n, c, out_h, out_w, kh, kw),
        strides=(sn, sc, sy * sh, sx * sw, sy * dh, sx * dw),
        writeable=False,
    )


# ----------------------------------------------------------------------
# convolutions
# ----------------------------------------------------------------------


def conv(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    *,
    strides=None,
    pads=None,
    dilations=None,
    group: int = 1,
    auto_pad: bytes = b"NOTSET",
) -> np.ndarray:
    """N-batch 2D convolution in NCHW layout."""
    if x.ndim != 4:
        raise OpError("Conv supports 4-D tensors only, got %dD" % x.ndim)
    x = np.ascontiguousarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)

    kh, kw = weight.shape[2], weight.shape[3]
    stride = _pairs(strides, 2, 1)
    dilation = _pairs(dilations, 2, 1)

    if auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
        top, bottom = _auto_pad_sizes(auto_pad, x.shape[2], kh, stride[0], dilation[0])
        left, right = _auto_pad_sizes(auto_pad, x.shape[3], kw, stride[1], dilation[1])
        pad = [top, left, bottom, right]
    else:
        raw = _pairs(pads, 4, 0)
        pad = [raw[0], raw[1], raw[2], raw[3]]

    padded = _spatial_pad(x, pad)
    n, c, h, w = padded.shape
    out_h = (h - ((kh - 1) * dilation[0] + 1)) // stride[0] + 1
    out_w = (w - ((kw - 1) * dilation[1] + 1)) // stride[1] + 1
    if out_h <= 0 or out_w <= 0:
        raise OpError("convolution output would be empty")

    out_channels = weight.shape[0]
    out = np.empty((n, out_channels, out_h, out_w), dtype=np.float32)
    in_per_group = c // group
    out_per_group = out_channels // group

    # Process a block of output rows at a time so the patch matrix stays small.
    row_bytes = max(1, out_w * in_per_group * kh * kw * 4)
    block = max(1, min(out_h, CONV_BLOCK_BYTES // row_bytes))

    for g in range(group):
        weight_g = weight[g * out_per_group : (g + 1) * out_per_group].reshape(out_per_group, -1)
        source = padded[:, g * in_per_group : (g + 1) * in_per_group]
        for start in range(0, out_h, block):
            stop = min(out_h, start + block)
            rows = stop - start
            window = source[:, :, start * stride[0] : (stop - 1) * stride[0] + (kh - 1) * dilation[0] + 1]
            patch = _patches(window, (kh, kw), stride, dilation, rows, out_w)
            flat = patch.transpose(0, 2, 3, 1, 4, 5).reshape(n * rows * out_w, -1)
            result = flat @ weight_g.T
            out[:, g * out_per_group : (g + 1) * out_per_group, start:stop] = (
                result.reshape(n, rows, out_w, out_per_group).transpose(0, 3, 1, 2)
            )

    if bias is not None:
        out += np.asarray(bias, dtype=np.float32).reshape(1, -1, 1, 1)
    return out


def conv_transpose(
    x: np.ndarray,
    weight: np.ndarray,
    bias: Optional[np.ndarray] = None,
    *,
    strides=None,
    pads=None,
    dilations=None,
    group: int = 1,
    output_padding=None,
    auto_pad: bytes = b"NOTSET",
) -> np.ndarray:
    """Transposed convolution - the upsampling half of the network."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)  # (Cin, Cout/group, kh, kw)
    stride = _pairs(strides, 2, 1)
    dilation = _pairs(dilations, 2, 1)
    pad = _pairs(pads, 4, 0)
    extra = _pairs(output_padding, 2, 0)

    n, c, h, w = x.shape
    kh, kw = weight.shape[2], weight.shape[3]
    out_per_group = weight.shape[1]
    out_channels = out_per_group * group
    in_per_group = c // group

    full_h = (h - 1) * stride[0] + (kh - 1) * dilation[0] + 1 + extra[0]
    full_w = (w - 1) * stride[1] + (kw - 1) * dilation[1] + 1 + extra[1]
    accumulator = np.zeros((n, out_channels, full_h, full_w), dtype=np.float32)

    for g in range(group):
        x_g = x[:, g * in_per_group : (g + 1) * in_per_group]
        w_g = weight[g * in_per_group : (g + 1) * in_per_group]
        # (N, Cout, kh, kw, H, W) contributions, accumulated per kernel offset.
        spread = np.einsum("nchw,cokl->nokhlw", x_g, w_g, optimize=True)
        for ky in range(kh):
            for kx in range(kw):
                y0 = ky * dilation[0]
                x0 = kx * dilation[1]
                accumulator[
                    :,
                    g * out_per_group : (g + 1) * out_per_group,
                    y0 : y0 + (h - 1) * stride[0] + 1 : stride[0],
                    x0 : x0 + (w - 1) * stride[1] + 1 : stride[1],
                ] += spread[:, :, ky, :, kx, :]

    top, left, bottom, right = pad
    out = accumulator[
        :,
        :,
        top : full_h - bottom if bottom else full_h,
        left : full_w - right if right else full_w,
    ]
    if bias is not None:
        out = out + np.asarray(bias, dtype=np.float32).reshape(1, -1, 1, 1)
    return np.ascontiguousarray(out)


# ----------------------------------------------------------------------
# pooling
# ----------------------------------------------------------------------


def _pool(x, kernel, strides, pads, dilations, auto_pad, ceil_mode, reducer, pad_value):
    """Shared implementation of MaxPool and AveragePool over the trailing axes."""
    kh, kw = kernel
    stride = _pairs(strides, 2, 1)
    dilation = _pairs(dilations, 2, 1)

    if auto_pad in (b"SAME_UPPER", b"SAME_LOWER"):
        top, bottom = _auto_pad_sizes(auto_pad, x.shape[2], kh, stride[0], dilation[0])
        left, right = _auto_pad_sizes(auto_pad, x.shape[3], kw, stride[1], dilation[1])
        pad_sizes = [top, left, bottom, right]
    else:
        raw = _pairs(pads, 4, 0)
        pad_sizes = [raw[0], raw[1], raw[2], raw[3]]

    if any(pad_sizes):
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_sizes[0], pad_sizes[2]), (pad_sizes[1], pad_sizes[3])),
            mode="constant",
            constant_values=pad_value,
        )
    else:
        padded = np.ascontiguousarray(x)

    effective_h = (kh - 1) * dilation[0] + 1
    effective_w = (kw - 1) * dilation[1] + 1
    round_up = np.ceil if ceil_mode else np.floor
    out_h = int(round_up((padded.shape[2] - effective_h) / float(stride[0]))) + 1
    out_w = int(round_up((padded.shape[3] - effective_w) / float(stride[1]))) + 1

    # ceil_mode may ask for one window past the edge; repeat the border for it.
    needed_h = (out_h - 1) * stride[0] + effective_h
    needed_w = (out_w - 1) * stride[1] + effective_w
    if needed_h > padded.shape[2] or needed_w > padded.shape[3]:
        padded = np.pad(
            padded,
            ((0, 0), (0, 0), (0, max(0, needed_h - padded.shape[2])), (0, max(0, needed_w - padded.shape[3]))),
            mode="edge",
        )

    padded = np.ascontiguousarray(padded)
    patches = _patches(padded, (kh, kw), stride, dilation, out_h, out_w)
    return reducer(patches, axis=(4, 5))


def max_pool(
    x,
    *,
    kernel_shape,
    strides=None,
    pads=None,
    dilations=None,
    auto_pad: bytes = b"NOTSET",
    ceil_mode: int = 0,
):
    return _pool(
        np.asarray(x, dtype=np.float32), kernel_shape, strides, pads, dilations,
        auto_pad, ceil_mode, np.max, -np.inf,
    )


def average_pool(
    x,
    *,
    kernel_shape,
    strides=None,
    pads=None,
    dilations=None,
    auto_pad: bytes = b"NOTSET",
    ceil_mode: int = 0,
):
    return _pool(
        np.asarray(x, dtype=np.float32), kernel_shape, strides, pads, dilations,
        auto_pad, ceil_mode, np.mean, 0.0,
    )


# ----------------------------------------------------------------------
# normalisation and activations
# ----------------------------------------------------------------------


def batch_normalization(x, scale, bias, mean, variance, *, epsilon: float = 1e-5):
    shape = [1] * x.ndim
    shape[1] = -1
    scale = np.asarray(scale, dtype=np.float32).reshape(shape)
    bias = np.asarray(bias, dtype=np.float32).reshape(shape)
    mean = np.asarray(mean, dtype=np.float32).reshape(shape)
    variance = np.asarray(variance, dtype=np.float32).reshape(shape)
    return (x - mean) / np.sqrt(variance + epsilon) * scale + bias


def instance_normalization(x, scale, bias, *, epsilon: float = 1e-5):
    axes = tuple(range(2, x.ndim))
    mean = x.mean(axis=axes, keepdims=True)
    variance = x.var(axis=axes, keepdims=True)
    shape = [1] * x.ndim
    shape[1] = -1
    normalised = (x - mean) / np.sqrt(variance + epsilon)
    return normalised * np.asarray(scale, np.float32).reshape(shape) + np.asarray(
        bias, np.float32
    ).reshape(shape)


def relu(x):
    return np.maximum(x, 0)


def leaky_relu(x, *, alpha: float = 0.01):
    return np.where(x >= 0, x, x * alpha)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0), dtype=np.float32))


def softmax(x, *, axis: int = -1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


# ----------------------------------------------------------------------
# shape plumbing
# ----------------------------------------------------------------------


def pad(x, pads: Sequence[int], value: float = 0.0, mode: bytes = b"constant", axes=None):
    rank = x.ndim
    pads = [int(item) for item in np.asarray(pads).ravel()] if pads is not None else []
    if not pads:
        # Exporters emit dynamically computed pads that can come out empty when
        # the input already has the required size; that is a no-op, not an error.
        return x
    if axes is None:
        axes = list(range(len(pads) // 2))
    else:
        axes = [int(axis) % rank for axis in axes]
    widths = [[0, 0] for _ in range(rank)]
    half = len(pads) // 2
    for index, axis in enumerate(axes):
        widths[axis][0] = int(pads[index])
        widths[axis][1] = int(pads[index + half])

    if any(before < 0 or after < 0 for before, after in widths):
        slices = []
        for axis, (before, after) in enumerate(widths):
            start = max(0, -before)
            stop = x.shape[axis] - max(0, -after)
            slices.append(slice(start, stop))
            widths[axis] = [max(0, before), max(0, after)]
        x = x[tuple(slices)]
    if not any(before or after for before, after in widths):
        return x

    name = bytes(mode).decode() if isinstance(mode, (bytes, bytearray)) else str(mode)
    if name == "constant":
        return np.pad(x, widths, mode="constant", constant_values=value)
    if name == "reflect":
        return np.pad(x, widths, mode="reflect")
    if name == "edge":
        return np.pad(x, widths, mode="edge")
    if name == "wrap":
        return np.pad(x, widths, mode="wrap")
    raise OpError("unsupported Pad mode %r" % name)


def slice_op(x, starts, ends, axes=None, steps=None):
    """ONNX Slice, including reversed (negative step) slices.

    The index conventions differ from Python's in one important place: with a
    negative step, an ``end`` below the first element means "run past the
    beginning", while Python would read a negative stop as counting from the
    end.  Getting that wrong silently produces empty tensors.
    """
    rank = x.ndim
    if axes is None:
        axes = list(range(len(starts)))
    if steps is None:
        steps = [1] * len(starts)
    slices = [slice(None)] * rank
    for index, axis in enumerate(axes):
        axis = int(axis) % rank
        step = int(steps[index])
        if step == 0:
            raise OpError("Slice step must not be zero")
        limit = x.shape[axis]
        start = int(starts[index])
        end = int(ends[index])
        start = start + limit if start < 0 else start
        end = end + limit if end < 0 else end
        if step > 0:
            start = min(max(start, 0), limit)
            end = min(max(end, 0), limit)
            slices[axis] = slice(start, end, step)
        else:
            start = min(max(start, -1), limit - 1)
            end = min(max(end, -1), limit - 1)
            # stop=None reaches element 0; a plain -1 would mean "the last one".
            slices[axis] = slice(start, None if end < 0 else end, step)
    return x[tuple(slices)]


def gather(x, indices, *, axis: int = 0):
    return np.take(x, np.asarray(indices, dtype=np.int64), axis=axis)


def gather_elements(x, indices, *, axis: int = 0):
    return np.take_along_axis(x, np.asarray(indices, dtype=np.int64), axis=axis)


def reshape(x, shape, *, allowzero: int = 0):
    target = [int(value) for value in np.asarray(shape).ravel()]
    if not allowzero:
        target = [x.shape[index] if value == 0 else value for index, value in enumerate(target)]
    return x.reshape(target)


def transpose(x, perm=None):
    return np.transpose(x, axes=perm if perm else None)


def unsqueeze(x, axes):
    result = x
    for axis in sorted(int(a) % (result.ndim + 1) for a in np.asarray(axes).ravel()):
        result = np.expand_dims(result, axis)
    return result


def squeeze(x, axes=None):
    if axes is None:
        return np.squeeze(x)
    return np.squeeze(x, axis=tuple(int(a) % x.ndim for a in np.asarray(axes).ravel()))


def constant_of_shape(shape, value=None):
    dims = [int(item) for item in np.asarray(shape).ravel()]
    fill = np.float32(0) if value is None else np.asarray(value).ravel()[0]
    return np.full(dims, fill, dtype=np.asarray(fill).dtype)


def range_op(start, limit, delta):
    return np.arange(np.asarray(start).item(), np.asarray(limit).item(), np.asarray(delta).item())


ONNX_TO_NUMPY = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
    12: np.uint32,
    13: np.uint64,
}


def cast(x, *, to: int = 1):
    dtype = ONNX_TO_NUMPY.get(int(to))
    if dtype is None:
        raise OpError("unsupported Cast target %d" % to)
    return np.asarray(x).astype(dtype)


def resize_nearest_or_linear(x, scales=None, sizes=None, mode: bytes = b"nearest"):
    """Resize the two trailing axes; enough for the upsampling used here."""
    if sizes is not None and len(np.asarray(sizes).ravel()):
        target = [int(v) for v in np.asarray(sizes).ravel()]
        out_h, out_w = target[-2], target[-1]
    elif scales is not None:
        factors = [float(v) for v in np.asarray(scales).ravel()]
        out_h = int(round(x.shape[-2] * factors[-2]))
        out_w = int(round(x.shape[-1] * factors[-1]))
    else:
        raise OpError("Resize needs either scales or sizes")

    name = bytes(mode).decode() if isinstance(mode, (bytes, bytearray)) else str(mode)
    rows = np.minimum((np.arange(out_h) * x.shape[-2] // max(1, out_h)), x.shape[-2] - 1)
    cols = np.minimum((np.arange(out_w) * x.shape[-1] // max(1, out_w)), x.shape[-1] - 1)
    if name == "nearest":
        return x[..., rows, :][..., :, cols]
    if name in ("linear", "bilinear"):
        y = (np.arange(out_h) + 0.5) * x.shape[-2] / out_h - 0.5
        xx = (np.arange(out_w) + 0.5) * x.shape[-1] / out_w - 0.5
        y0 = np.clip(np.floor(y).astype(int), 0, x.shape[-2] - 1)
        y1 = np.clip(y0 + 1, 0, x.shape[-2] - 1)
        x0 = np.clip(np.floor(xx).astype(int), 0, x.shape[-1] - 1)
        x1 = np.clip(x0 + 1, 0, x.shape[-1] - 1)
        wy = np.clip(y - y0, 0, 1).reshape(-1, 1)
        wx = np.clip(xx - x0, 0, 1).reshape(1, -1)
        top = x[..., y0, :][..., :, x0] * (1 - wx) + x[..., y0, :][..., :, x1] * wx
        bottom = x[..., y1, :][..., :, x0] * (1 - wx) + x[..., y1, :][..., :, x1] * wx
        return top * (1 - wy) + bottom * wy
    raise OpError("unsupported Resize mode %r" % name)
