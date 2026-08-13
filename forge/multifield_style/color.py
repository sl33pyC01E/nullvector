from __future__ import annotations

import math
from typing import Iterable

import numpy as np


RGB = tuple[int, int, int]
OKLAB = tuple[float, float, float]


def _linear_to_srgb(channel: float) -> float:
    if channel <= 0.0031308:
        return 12.92 * channel
    return 1.055 * (max(channel, 0.0) ** (1.0 / 2.4)) - 0.055


def _srgb_to_linear(channel: float) -> float:
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _oklab_to_linear_rgb(lab: OKLAB) -> tuple[float, float, float]:
    lightness, axis_a, axis_b = lab
    l_root = lightness + 0.3963377774 * axis_a + 0.2158037573 * axis_b
    m_root = lightness - 0.1055613458 * axis_a - 0.0638541728 * axis_b
    s_root = lightness - 0.0894841775 * axis_a - 1.2914855480 * axis_b
    l_value = l_root * l_root * l_root
    m_value = m_root * m_root * m_root
    s_value = s_root * s_root * s_root
    return (
        4.0767416621 * l_value - 3.3077115913 * m_value + 0.2309699292 * s_value,
        -1.2684380046 * l_value + 2.6097574011 * m_value - 0.3413193965 * s_value,
        -0.0041960863 * l_value - 0.7034186147 * m_value + 1.7076147010 * s_value,
    )


def oklch_to_srgb8(lightness: float, chroma: float, hue_degrees: float) -> RGB:
    """Convert OKLCH to bounded sRGB with deterministic chroma clipping.

    Lightness is clamped to the displayable interval. Chroma is reduced with
    a fixed-iteration binary search when the requested color lies outside the
    sRGB gamut. No color-management dependency or platform profile is involved.
    """

    lightness = min(0.98, max(0.02, float(lightness)))
    chroma = min(0.37, max(0.0, float(chroma)))
    radians = math.radians(float(hue_degrees) % 360.0)

    def candidate(current_chroma: float) -> tuple[float, float, float]:
        lab = (
            lightness,
            current_chroma * math.cos(radians),
            current_chroma * math.sin(radians),
        )
        return _oklab_to_linear_rgb(lab)

    linear = candidate(chroma)
    if any(channel < 0.0 or channel > 1.0 for channel in linear):
        low, high = 0.0, chroma
        for _ in range(24):
            middle = (low + high) * 0.5
            probe = candidate(middle)
            if all(0.0 <= channel <= 1.0 for channel in probe):
                low = middle
            else:
                high = middle
        linear = candidate(low)
    encoded = [_linear_to_srgb(min(1.0, max(0.0, channel))) for channel in linear]
    return tuple(int(min(255, max(0, math.floor(channel * 255.0 + 0.5)))) for channel in encoded)  # type: ignore[return-value]


def srgb8_to_oklab(rgb: Iterable[int] | np.ndarray) -> OKLAB:
    red, green, blue = (float(channel) / 255.0 for channel in rgb)
    red, green, blue = (
        _srgb_to_linear(red),
        _srgb_to_linear(green),
        _srgb_to_linear(blue),
    )
    l_value = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    m_value = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    s_value = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    l_root = math.copysign(abs(l_value) ** (1.0 / 3.0), l_value)
    m_root = math.copysign(abs(m_value) ** (1.0 / 3.0), m_value)
    s_root = math.copysign(abs(s_value) ** (1.0 / 3.0), s_value)
    return (
        0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root,
        1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root,
        0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root,
    )


def delta_e_oklab(first: Iterable[int], second: Iterable[int]) -> float:
    first_lab = srgb8_to_oklab(first)
    second_lab = srgb8_to_oklab(second)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first_lab, second_lab)))


def rgb_array_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB uint8 -> OKLab conversion for small metric arrays."""

    values = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    red, green, blue = linear[..., 0], linear[..., 1], linear[..., 2]
    l_value = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    m_value = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    s_value = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    l_root, m_root, s_root = np.cbrt(l_value), np.cbrt(m_value), np.cbrt(s_value)
    return np.stack(
        (
            0.2104542553 * l_root + 0.7936177850 * m_root - 0.0040720468 * s_root,
            1.9779984951 * l_root - 2.4285922050 * m_root + 0.4505937099 * s_root,
            0.0259040371 * l_root + 0.7827717662 * m_root - 0.8086757660 * s_root,
        ),
        axis=-1,
    )
