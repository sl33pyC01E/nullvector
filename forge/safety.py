from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any

from .config import PROJECT_ROOT


GIB = 1024**3
DEFAULT_DISK_FLOOR_GB = 100.0


@dataclass(frozen=True, slots=True)
class DiskStatus:
    root: str
    total_gb: float
    used_gb: float
    free_gb: float
    floor_gb: float
    planned_gb: float
    safe: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def disk_status(
    path: Path = PROJECT_ROOT,
    *,
    floor_gb: float = DEFAULT_DISK_FLOOR_GB,
    planned_bytes: int = 0,
) -> DiskStatus:
    path = Path(path).resolve()
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_after_plan = usage.free - max(int(planned_bytes), 0)
    return DiskStatus(
        root=path.anchor,
        total_gb=round(usage.total / GIB, 3),
        used_gb=round(usage.used / GIB, 3),
        free_gb=round(usage.free / GIB, 3),
        floor_gb=float(floor_gb),
        planned_gb=round(max(int(planned_bytes), 0) / GIB, 3),
        safe=free_after_plan >= floor_gb * GIB,
    )


def require_disk_floor(
    path: Path = PROJECT_ROOT,
    *,
    floor_gb: float = DEFAULT_DISK_FLOOR_GB,
    planned_bytes: int = 0,
) -> DiskStatus:
    status = disk_status(path, floor_gb=floor_gb, planned_bytes=planned_bytes)
    if not status.safe:
        raise RuntimeError(
            "Disk safety floor reached: "
            f"{status.free_gb:.3f} GiB free, {status.planned_gb:.3f} GiB planned, "
            f"{status.floor_gb:.3f} GiB must remain. Stop generation/training."
        )
    return status


def write_json_atomic(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_disk_floor(path.parent, planned_bytes=256 * 1024)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
