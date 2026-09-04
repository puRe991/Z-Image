"""Engine registry."""

from __future__ import annotations

from typing import Any, Dict, List

from .base import Cancelled, Engine, EngineError, GenerationContext

#: Name -> import path of the engine class.  The Z-Image engine is imported
#: lazily so that the demo engine still works on a machine without PyTorch.
_ENGINES = {
    "mock": ("zimage_studio.server.engines.mock", "MockEngine"),
    "zimage": ("zimage_studio.server.engines.zimage_local", "ZImageEngine"),
}


def list_engines() -> List[str]:
    return sorted(_ENGINES)


def create_engine(name: str, **kwargs: Any) -> Engine:
    """Instantiate an engine by name."""
    import importlib

    try:
        module_name, class_name = _ENGINES[name]
    except KeyError:
        raise EngineError("unknown engine %r (available: %s)" % (name, ", ".join(list_engines()))) from None
    module = importlib.import_module(module_name)
    return getattr(module, class_name)(**kwargs)


__all__ = ["Cancelled", "Engine", "EngineError", "GenerationContext", "create_engine", "list_engines"]
