# Model files

This directory holds the neural network used by the **local** engine.  The
weights are not part of the repository (they are 204 MB); the file is fetched
separately.

## lama_fp32.onnx

|  |  |
|---|---|
| **What** | LaMa - resolution-robust large-mask inpainting with Fourier convolutions |
| **Size** | 204 MB, 51 million parameters, fixed 512×512 input |
| **Source** | https://huggingface.co/Carve/LaMa-ONNX → `lama_fp32.onnx` |
| **Upstream** | https://github.com/advimman/lama (Samsung AI Center) |
| **License** | Apache-2.0 |
| **SHA-256** | `1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6` |

Download it with:

```bash
python tools/build_offline_bundle.py --out ZImageStudio-Offline
```

or by hand into this directory.  The application also accepts the file
elsewhere: set it under *Processing → Choose AI model file…*, or point the
environment variable `ZIMAGE_LAMA_MODEL` at it.

Without the file the application still runs; object removal then uses the
dependency-free content-aware fill instead, which is smoother and less
convincing on textured backgrounds.

## u2netp.onnx

|  |  |
|---|---|
| **What** | U²-Net (small variant) - salient object detection, used for "cut out" and the background operations |
| **Size** | 4.5 MB, 1.1 million parameters, fixed 320×320 input |
| **Source** | https://github.com/danielgatis/rembg/releases → `u2netp.onnx` |
| **Upstream** | https://github.com/xuebinqin/U-2-Net |
| **License** | Apache-2.0 |
| **SHA-256** | `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8` |

Found automatically in this directory, or pointed at with the environment
variable `ZIMAGE_U2NET_MODEL`.
