"""Z-Image Studio - image-to-image editing suite with a 32-bit Windows GUI client.

The package is split in two halves:

* :mod:`zimage_studio.client` - a Tkinter desktop application that only uses the
  Python standard library, so it runs on a 32-bit Windows Python installation
  where PyTorch has no wheels.
* :mod:`zimage_studio.server` - an HTTP service that owns the Z-Image weights and
  performs txt2img / img2img / inpaint on a 64-bit host (or the same machine).

:mod:`zimage_studio.protocol` and :mod:`zimage_studio.imaging` are shared by both
and must stay dependency free.
"""

from .version import __version__

__all__ = ["__version__"]
