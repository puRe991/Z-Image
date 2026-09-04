"""Image-to-image and inpainting maths of the Z-Image sampling loop.

The real weights need a GPU and 12 GB of downloads, so the transformer, text
encoder and VAE are replaced by stubs.  What is exercised here is exactly the
code this feature added: where the sampler starts, how the source latents are
noised, and how the mask re-anchors the kept region.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from zimage.pipeline import generate  # noqa: E402
from zimage.scheduler import FlowMatchEulerDiscreteScheduler  # noqa: E402

LATENT_CHANNELS = 4
SIZE = 64  # -> 8x8 latents


class _Tokenizer:
    model_max_length = 32

    def apply_chat_template(self, messages, **kwargs):
        return messages[0]["content"]

    def __call__(self, prompts, padding=None, max_length=8, truncation=True, return_tensors="pt"):
        batch = len(prompts)
        length = 4

        class _Encoded:
            input_ids = torch.zeros(batch, length, dtype=torch.long)
            attention_mask = torch.ones(batch, length, dtype=torch.long)

        return _Encoded()


class _TextEncoderOutput:
    def __init__(self, hidden):
        self.hidden_states = [hidden, hidden, hidden]


class _TextEncoder:
    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=True):
        batch, length = input_ids.shape
        return _TextEncoderOutput(torch.zeros(batch, length, 16))


class _Transformer(torch.nn.Module):
    """Predicts a constant velocity so the trajectory is exactly predictable."""

    in_channels = LATENT_CHANNELS

    def __init__(self, value: float = 0.0, random: bool = False):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros(1))
        self.value = value
        self.random = random
        self.calls = 0

    def forward(self, latents, timestep, prompt_embeds):
        self.calls += 1
        if self.random:
            out = [torch.randn_like(item) for item in latents]
        else:
            out = [torch.full_like(item, self.value) for item in latents]
        return (out,)


class _VaeConfig:
    block_out_channels = (1, 2, 3, 4)  # len - 1 = 3 -> /8
    scaling_factor = 1.0
    shift_factor = 0.0


class _Vae:
    config = _VaeConfig()
    dtype = torch.float32


def _components(transformer=None):
    return {
        "transformer": transformer or _Transformer(),
        "vae": _Vae(),
        "text_encoder": _TextEncoder(),
        "tokenizer": _Tokenizer(),
        "scheduler": FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0),
    }


def _latents(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, LATENT_CHANNELS, SIZE // 8, SIZE // 8, generator=generator)


def _run(**kwargs):
    params = dict(
        prompt="a test",
        height=SIZE,
        width=SIZE,
        num_inference_steps=6,
        guidance_scale=0.0,
        generator=torch.Generator().manual_seed(1234),
        output_type="latent",
    )
    params.update(kwargs)
    components = params.pop("components", None) or _components()
    return generate(**components, **params), components


def test_txt2img_still_works_without_the_new_arguments():
    latents, _ = _run()
    assert latents.shape == (1, LATENT_CHANNELS, SIZE // 8, SIZE // 8)
    assert latents.dtype == torch.float32


def test_lower_strength_stays_closer_to_the_source():
    source = _latents()
    distances = []
    for strength in (0.2, 0.5, 0.9):
        latents, _ = _run(init_latents=source, strength=strength)
        distances.append(float((latents - source).abs().mean()))
    assert distances[0] < distances[1] < distances[2]


def test_strength_one_starts_from_pure_noise():
    """At strength 1.0 the source must not influence the starting point."""
    source = _latents() * 50.0
    latents, _ = _run(init_latents=source, strength=1.0)
    plain, _ = _run()
    assert torch.allclose(latents, plain, atol=1e-5)


def test_img2img_runs_fewer_steps_than_txt2img():
    source = _latents()
    _, components = _run(init_latents=source, strength=0.5, num_inference_steps=10)
    assert components["transformer"].calls == 5


def test_zero_mask_reproduces_the_source_exactly():
    """Everything outside the brushed area has to survive untouched."""
    source = _latents(3)
    mask = torch.zeros(1, 1, SIZE // 8, SIZE // 8)
    latents, _ = _run(
        components=_components(_Transformer(random=True)),
        init_latents=source,
        strength=1.0,
        mask_latents=mask,
    )
    assert torch.allclose(latents, source, atol=1e-5)


def test_full_mask_behaves_like_plain_img2img():
    source = _latents(4)
    ones = torch.ones(1, 1, SIZE // 8, SIZE // 8)
    masked, _ = _run(init_latents=source, strength=0.6, mask_latents=ones)
    plain, _ = _run(init_latents=source, strength=0.6)
    assert torch.allclose(masked, plain, atol=1e-5)


def test_partial_mask_only_changes_the_masked_region():
    source = _latents(5)
    mask = torch.zeros(1, 1, SIZE // 8, SIZE // 8)
    mask[..., :4, :] = 1.0  # top half is repainted
    latents, _ = _run(
        components=_components(_Transformer(random=True)),
        init_latents=source,
        strength=1.0,
        mask_latents=mask,
    )
    assert torch.allclose(latents[..., 4:, :], source[..., 4:, :], atol=1e-5)
    assert not torch.allclose(latents[..., :4, :], source[..., :4, :], atol=1e-3)


def test_mask_without_source_is_rejected():
    with pytest.raises(ValueError, match="init_latents"):
        _run(mask_latents=torch.ones(1, 1, SIZE // 8, SIZE // 8))


def test_mismatched_source_shape_is_rejected():
    with pytest.raises(ValueError, match="init_latents shape"):
        _run(init_latents=torch.zeros(1, LATENT_CHANNELS, 4, 4))


def test_mismatched_mask_grid_is_rejected():
    with pytest.raises(ValueError, match="mask_latents grid"):
        _run(init_latents=_latents(), mask_latents=torch.ones(1, 1, 3, 3))


def test_callback_reports_every_step():
    seen = []
    _run(num_inference_steps=5, callback=lambda step, total: seen.append((step, total)))
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]


def test_callback_can_abort_the_run():
    class _Stop(Exception):
        pass

    def callback(step, total):
        if step == 2:
            raise _Stop()

    with pytest.raises(_Stop):
        _run(num_inference_steps=8, callback=callback)


def test_source_is_broadcast_over_multiple_images():
    source = _latents(7)
    latents, _ = _run(init_latents=source, strength=0.4, num_images_per_prompt=3)
    assert latents.shape[0] == 3


# ----------------------------------------------------------------------
# VAE encode (added for image-to-image)
# ----------------------------------------------------------------------


def _tiny_vae():
    from zimage.autoencoder import AutoencoderKL

    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        block_out_channels=(8, 8, 8, 8),  # three downsamples -> /8
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        scaling_factor=0.5,
        shift_factor=0.25,
    )


def test_vae_encode_returns_a_latent_distribution():
    vae = _tiny_vae().eval()
    image = torch.zeros(1, 3, 32, 32)
    with torch.no_grad():
        posterior = vae.encode(image).latent_dist
    assert posterior.mode().shape == (1, 4, 4, 4)
    assert posterior.std.shape == posterior.mean.shape
    assert torch.equal(posterior.mode(), posterior.mean)


def test_vae_encode_sampling_is_reproducible():
    vae = _tiny_vae().eval()
    image = torch.rand(1, 3, 32, 32) * 2 - 1
    with torch.no_grad():
        posterior = vae.encode(image).latent_dist
        first = posterior.sample(torch.Generator().manual_seed(5))
        second = posterior.sample(torch.Generator().manual_seed(5))
        third = posterior.sample(torch.Generator().manual_seed(6))
    assert torch.allclose(first, second)
    assert not torch.allclose(first, third)


def test_vae_encode_decode_shapes_line_up():
    """Latents produced by encode() must be decodable back to the input size."""
    vae = _tiny_vae().eval()
    image = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.mode()
        decoded = vae.decode(latents, return_dict=False)[0]
    assert decoded.shape == image.shape


def test_scaled_latents_round_trip_through_the_pipeline_convention():
    """The engine scales latents the way the pipeline unscales them."""
    vae = _tiny_vae().eval()
    with torch.no_grad():
        raw = vae.encode(torch.zeros(1, 3, 32, 32)).latent_dist.mode()
    shift = vae.config.shift_factor
    scaled = (raw - shift) * vae.config.scaling_factor          # engine side
    unscaled = (scaled / vae.config.scaling_factor) + shift      # pipeline side
    assert torch.allclose(raw, unscaled, atol=1e-6)
