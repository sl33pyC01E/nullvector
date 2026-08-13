from __future__ import annotations


MASK64 = (1 << 64) - 1


def mix64(value: int) -> int:
    """SplitMix64 finalizer with explicit unsigned wrapping."""
    value = (int(value) + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return (value ^ (value >> 31)) & MASK64


def coordinate_hash(seed: int, x: int, y: int, salt: int = 0) -> int:
    """Order-independent random access into a deterministic 64-bit field."""
    value = int(seed) & MASK64
    value ^= (int(x) * 0xD6E8FEB86659FD93) & MASK64
    value ^= (int(y) * 0xA5A3564E27F8862D) & MASK64
    value ^= (int(salt) * 0x9E3779B97F4A7C15) & MASK64
    return mix64(value)


def bounded_hash(seed: int, x: int, y: int, salt: int, bound: int) -> int:
    if bound < 1:
        raise ValueError("bound must be positive")
    return coordinate_hash(seed, x, y, salt) % bound

