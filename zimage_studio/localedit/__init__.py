"""Local, model-free image editing.

Everything in here runs on the machine that shows the user interface - no
network, no GPU, no compiled dependencies.  It is what makes the application
useful on a 32-bit Windows laptop that cannot host the Z-Image weights.

It is classical image processing, not a neural network: adjustments are
lookup-table operations, and "replace area" fills the brushed region from its
surroundings.  It cannot invent content that was never in the picture.
"""

from .adjust import (
    apply_operation,
    OPERATIONS,
)
from .commands import Command, describe_vocabulary, parse_prompt
from .fill import content_aware_fill

__all__ = [
    "Command",
    "OPERATIONS",
    "apply_operation",
    "content_aware_fill",
    "describe_vocabulary",
    "parse_prompt",
]
