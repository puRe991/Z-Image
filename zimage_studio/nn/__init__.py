"""A dependency-light neural network runtime.

Reading ONNX files needs nothing but the standard library; executing them needs
NumPy, which - unlike PyTorch or onnxruntime - still has 32-bit Windows wheels
(up to NumPy 1.24 for CPython 3.8 - 3.11).

That combination is what lets this application run a real neural network on the
kind of machine it targets.
"""

from .onnx_model import Graph, OnnxError, load

__all__ = ["Graph", "OnnxError", "load"]
