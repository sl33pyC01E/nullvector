from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import shutil

from .constants import CANVAS_SIZE, LAYER_NAMES


DEFAULT_RESERVE_BYTES = 2 * 1024**3
DEFAULT_MAX_SAMPLES = 250_000
RAW_BYTES_PER_SAMPLE = (
    len(LAYER_NAMES) * CANVAS_SIZE * CANVAS_SIZE
    + CANVAS_SIZE * CANVAS_SIZE
    + 12_288
)


class DiskBudgetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class DiskBudget:
    sample_count: int
    estimated_bytes: int
    free_bytes: int
    reserve_bytes: int
    writable_bytes: int
    bytes_per_sample: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def estimate_corpus_bytes(
    sample_count: int,
    *,
    bytes_per_sample: int = RAW_BYTES_PER_SAMPLE,
    headroom: float = 1.25,
) -> int:
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 0:
        raise ValueError("sample_count must be a non-negative integer")
    if bytes_per_sample <= 0 or headroom < 1.0:
        raise ValueError("disk estimate parameters are invalid")
    return int(np_ceil(sample_count * bytes_per_sample * headroom))


def np_ceil(value: float) -> int:
    # Avoid importing NumPy in the disk guard used before corpus dependencies load.
    integer = int(value)
    return integer if value == integer else integer + 1


def plan_disk_budget(
    sample_count: int,
    *,
    free_bytes: int,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    bytes_per_sample: int = RAW_BYTES_PER_SAMPLE,
    allow_large: bool = False,
) -> DiskBudget:
    if sample_count > max_samples and not allow_large:
        raise DiskBudgetError(
            f"Requested {sample_count:,} samples exceeds the guarded limit "
            f"of {max_samples:,}; pass allow_large=True only after review."
        )
    estimated = estimate_corpus_bytes(
        sample_count, bytes_per_sample=bytes_per_sample
    )
    writable = max(0, int(free_bytes) - int(reserve_bytes))
    if estimated > writable:
        raise DiskBudgetError(
            f"Corpus needs approximately {estimated:,} bytes but only "
            f"{writable:,} guarded bytes are writable."
        )
    return DiskBudget(
        sample_count=sample_count,
        estimated_bytes=estimated,
        free_bytes=int(free_bytes),
        reserve_bytes=int(reserve_bytes),
        writable_bytes=writable,
        bytes_per_sample=bytes_per_sample,
    )


def _existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise DiskBudgetError(f"No existing parent found for {path}")
        candidate = candidate.parent
    return candidate


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def guard_corpus_destination(
    destination: Path,
    sample_count: int,
    *,
    reserve_bytes: int = DEFAULT_RESERVE_BYTES,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    allow_large: bool = False,
) -> DiskBudget:
    destination = Path(destination).resolve()
    project_root = Path(__file__).resolve().parents[2]
    protected = (
        project_root / "game",
        project_root / "checkpoints",
        project_root / "data",
    )
    if any(_is_relative_to(destination, path.resolve()) for path in protected):
        raise DiskBudgetError(
            "Broad morphology artifacts must remain isolated from production game, "
            "checkpoint, and v1 corpus paths."
        )
    usage = shutil.disk_usage(_existing_parent(destination))
    return plan_disk_budget(
        sample_count,
        free_bytes=usage.free,
        reserve_bytes=reserve_bytes,
        max_samples=max_samples,
        allow_large=allow_large,
    )
