"""Local editing without any dependency: pixel maths, adjustments, fill, prompts."""

from __future__ import annotations

import pytest

from zimage_studio.localedit import adjust, commands, fill, pixels


# ----------------------------------------------------------------------
# pixel primitives
# ----------------------------------------------------------------------


def test_split_and_merge_are_inverse():
    rgb = bytes(range(30))
    planes = pixels.split_planes(rgb)
    assert [len(plane) for plane in planes] == [10, 10, 10]
    assert bytes(pixels.merge_planes(planes)) == rgb


def test_neighbour_average_matches_the_naive_version():
    width, height = 7, 5
    plane = bytes((x * 13 + y * 29) % 256 for y in range(height) for x in range(width))
    result = pixels.neighbour_average(plane, width, height)

    def naive(x, y):
        up = plane[(y - 1) * width + x] if y > 0 else plane[y * width + x]
        down = plane[(y + 1) * width + x] if y < height - 1 else plane[y * width + x]
        left = plane[y * width + x - 1] if x > 0 else plane[y * width + x]
        right = plane[y * width + x + 1] if x < width - 1 else plane[y * width + x]
        return (up + down + left + right) // 4

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            assert result[y * width + x] == naive(x, y), (x, y)


def test_neighbour_average_does_not_wrap_across_rows():
    """A row's last pixel must not be averaged with the next row's first."""
    width, height = 4, 3
    plane = bytes([0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255])
    result = pixels.neighbour_average(plane, width, height)
    assert result[width - 1] > 0  # its own column and neighbours
    assert result[width] < 100  # the next row's first pixel stayed dark


def test_weighted_mix_matches_the_exact_formula():
    first = bytes([0, 50, 100, 200, 255])
    second = bytes([255, 200, 100, 50, 0])
    for amount in (0.0, 0.25, 0.5, 0.75, 1.0):
        mixed = pixels.weighted_mix(first, second, amount)
        expected = [int(a + (b - a) * amount) for a, b in zip(first, second)]
        assert max(abs(m - e) for m, e in zip(mixed, expected)) <= 2


def test_add_planes_sums_bytes():
    assert list(pixels.add_planes([bytes([10, 20]), bytes([5, 6]), bytes([1, 2])])) == [16, 28]


def test_down_and_upsample_keep_the_size_contract():
    width, height = 8, 6
    plane = bytes(range(width * height))
    small, small_w, small_h = pixels.downsample_plane(plane, width, height, 2)
    assert (small_w, small_h) == (4, 3) and len(small) == 12
    large, large_w, large_h = pixels.upsample_plane(small, small_w, small_h, 2, target=(width, height))
    assert (large_w, large_h) == (width, height) and len(large) == width * height


def test_crop_and_paste_roundtrip():
    width, height = 6, 4
    plane = bytearray(range(width * height))
    box = (1, 1, 4, 3)
    patch = pixels.crop_plane(bytes(plane), width, box)
    assert len(patch) == 3 * 2
    pixels.paste_plane(plane, width, box, bytes([9] * 6))
    assert plane[1 * width + 1] == 9 and plane[0] == 0


def test_make_table_clamps():
    table = pixels.make_table(lambda value: value * 4)
    assert table[0] == 0 and table[100] == 255


# ----------------------------------------------------------------------
# adjustments
# ----------------------------------------------------------------------


def _planes(value=(120, 100, 80), count=64):
    return [bytes([channel]) * count for channel in value]


def _mean(plane):
    return sum(plane) / float(len(plane))


def test_brighter_and_darker_move_the_right_way():
    planes = _planes()
    assert _mean(adjust.brighter(planes, 8, 8, 0.5)[0]) > _mean(planes[0])
    assert _mean(adjust.darker(planes, 8, 8, 0.5)[0]) < _mean(planes[0])


def test_contrast_pushes_away_from_and_towards_mid_grey():
    dark, bright = bytes([60]) * 16, bytes([200]) * 16
    planes = [dark, bright, dark]
    more = adjust.more_contrast(planes, 4, 4, 1.0)
    less = adjust.less_contrast(planes, 4, 4, 1.0)
    assert more[0][0] < dark[0] and more[1][0] > bright[0]
    assert abs(less[0][0] - 128) < abs(dark[0] - 128)


def test_grayscale_makes_all_channels_equal():
    result = adjust.grayscale(_planes(), 8, 8, 1.0)
    assert result[0] == result[1] == result[2]


def test_grayscale_uses_luma_weights():
    red = [bytes([255]), bytes([0]), bytes([0])]
    assert abs(adjust.grayscale(red, 1, 1, 1.0)[0][0] - 76) <= 1


def test_sepia_is_warmer_than_it_is_cool():
    result = adjust.sepia(_planes(), 8, 8, 1.0)
    assert result[0][0] > result[2][0]


def test_warmer_and_cooler_shift_the_channels():
    warm = adjust.warmer(_planes(), 8, 8, 1.0)
    cool = adjust.cooler(_planes(), 8, 8, 1.0)
    assert warm[0][0] > cool[0][0]
    assert cool[2][0] > warm[2][0]


def test_invert_twice_is_the_identity():
    planes = _planes()
    once = adjust.invert(planes, 8, 8)
    twice = adjust.invert(once, 8, 8)
    assert list(twice[0]) == list(planes[0])


def test_fill_color_replaces_everything():
    result = adjust.fill_color(_planes(), 8, 8, 1.0, color=(10, 20, 30))
    assert result[0][0] == 10 and result[1][0] == 20 and result[2][0] == 30


def test_blur_evens_out_a_hard_edge():
    width = height = 8
    plane = bytes([0] * 4 + [255] * 4) * height
    blurred = adjust.blur([plane] * 3, width, height, 0.8)[0]
    row = blurred[width * 4 : width * 5]
    assert 0 < row[3] < 255 or 0 < row[4] < 255


def test_unknown_operation_is_reported():
    with pytest.raises(KeyError):
        adjust.apply_operation("levitate", _planes(), 8, 8, 0.5)


# ----------------------------------------------------------------------
# content-aware fill
# ----------------------------------------------------------------------


def _scene(width=96, height=96):
    """A vertical gradient with a bright square in the middle."""
    plane = bytearray(width * height)
    for y in range(height):
        plane[y * width : (y + 1) * width] = bytes([(y * 255) // height]) * width
    mask = bytearray(width * height)
    for y in range(40, 56):
        for x in range(40, 56):
            plane[y * width + x] = 250
            mask[y * width + x] = 255
    return [bytes(plane)] * 3, bytes(mask), width, height


def test_fill_replaces_the_masked_area():
    planes, mask, width, height = _scene()
    filled = fill.content_aware_fill(planes, width, height, mask)
    centre = filled[0][48 * width + 48]
    assert abs(centre - (48 * 255) // height) < 40  # continues the gradient
    assert centre != 250  # the bright square is gone


def test_fill_leaves_everything_outside_untouched():
    planes, mask, width, height = _scene()
    filled = fill.content_aware_fill(planes, width, height, mask)
    for index in range(width * height):
        if not mask[index]:
            assert filled[0][index] == planes[0][index]


def test_fill_without_a_mask_is_a_no_op():
    planes, _mask, width, height = _scene()
    empty = bytes(width * height)
    assert fill.content_aware_fill(planes, width, height, empty)[0] == planes[0]


def test_bounding_box_covers_the_mask():
    _planes_, mask, width, height = _scene()
    box = fill.mask_bounding_box(mask, width, height)
    assert box == (40, 40, 56, 56)
    grown = fill.mask_bounding_box(mask, width, height, margin=5)
    assert grown == (35, 35, 61, 61)


def test_bounding_box_of_an_empty_mask_is_none():
    assert fill.mask_bounding_box(bytes(64), 8, 8) is None


# ----------------------------------------------------------------------
# prompt vocabulary
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("entferne das auto", ["remove"]),
        ("weg damit", ["remove"]),
        ("remove this", ["remove"]),
        ("etwas heller", ["brighter"]),
        ("viel dunkler", ["darker"]),
        ("mehr kontrast", ["more_contrast"]),
        ("weniger kontrast", ["less_contrast"]),
        ("schwarzweiß", ["grayscale"]),
        ("black and white", ["grayscale"]),
        ("sepia", ["sepia"]),
        ("wärmer", ["warmer"]),
        ("kühler", ["cooler"]),
        ("weichzeichnen", ["blur"]),
        ("schärfen", ["sharpen"]),
        ("invertieren", ["invert"]),
    ],
)
def test_single_instructions(prompt, expected):
    assert [command.name for command in commands.parse_prompt(prompt)] == expected


def test_combined_instructions_keep_their_order():
    parsed = commands.parse_prompt("entferne den fleck und mach es etwas heller")
    assert [command.name for command in parsed] == ["remove", "brighter"]
    assert all(abs(command.amount - 0.35) < 0.01 for command in parsed)


def test_intensity_words_scale_the_amount():
    assert commands.parse_prompt("leicht heller")[0].amount < commands.parse_prompt("heller")[0].amount
    assert commands.parse_prompt("sehr heller")[0].amount > commands.parse_prompt("heller")[0].amount


def test_fill_with_a_colour():
    parsed = commands.parse_prompt("füll das mit blau")
    assert parsed[0].name == "fill_color"
    assert parsed[0].options["color"] == adjust.COLORS["blue"]


def test_empty_prompt_with_a_mask_means_remove():
    assert [c.name for c in commands.parse_prompt("", has_mask=True)] == ["remove"]
    assert commands.parse_prompt("", has_mask=False) == []


def test_unknown_words_produce_nothing():
    assert commands.parse_prompt("a photorealistic lighthouse at dusk") == []


def test_vocabulary_help_is_available_in_both_languages():
    assert "entfernen" in commands.describe_vocabulary("de")
    assert "remove" in commands.describe_vocabulary("en")
