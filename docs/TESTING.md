# Tests

```bash
# everything except the GUI (no display needed)
python -m pytest tests --ignore=tests/test_gui.py

# the GUI integration tests need Tkinter and a display
python -m pytest tests/test_gui.py            # on a desktop
xvfb-run -a python -m pytest tests/test_gui.py  # headless / CI
```

| File | Covers |
|------|--------|
| `test_protocol.py` | request validation, clamping, size alignment, job/info parsing |
| `test_imaging.py` | pure-Python PNG codec, format sniffing, mask rasterising |
| `test_server_api.py` | the HTTP API end to end: generate, progress, cancel, auth, convert |
| `test_pipeline_img2img.py` | the sampling maths of image-to-image and inpainting, plus VAE encode |
| `test_gui.py` | the desktop client driving a real generation against the demo backend |

`test_pipeline_img2img.py` needs PyTorch (CPU is enough) but no model weights: the
transformer, text encoder and VAE are replaced by stubs, which is what makes the
strong invariants checkable - an all-zero mask has to reproduce the source
exactly, a full mask has to equal plain image-to-image, and strength 1.0 has to
equal text-to-image.

None of the tests download weights or need a GPU.
