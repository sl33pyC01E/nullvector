from __future__ import annotations

import hashlib
import os
from pathlib import Path
import uuid

import torch

from .contract import CHECKPOINT_FORMAT, source_sha256


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RecoveryCheckpointStore:
    """Atomic latest checkpoints plus immutable validation milestones."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.milestones = self.root / "milestones"

    def save(self, payload: dict, *, step: int, milestone: bool) -> dict:
        if step < 1 or payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
            raise ValueError("cellular recovery checkpoint provenance drifted")
        self.root.mkdir(parents=True, exist_ok=True)
        self.milestones.mkdir(exist_ok=True)
        temporary = self.root / f".checkpoint-{uuid.uuid4().hex}.tmp"
        torch.save(payload, temporary)
        digest = _file_sha256(temporary)
        latest = self.root / "latest.pt"
        os.replace(temporary, latest)
        milestone_path = None
        if milestone:
            milestone_path = self.milestones / f"step-{step:08d}.pt"
            if milestone_path.exists():
                raise FileExistsError(milestone_path)
            staging = self.milestones / f".{milestone_path.name}-{uuid.uuid4().hex}.tmp"
            staging.write_bytes(latest.read_bytes())
            os.replace(staging, milestone_path)
        return {"step": step, "latest": str(latest), "sha256": digest, "milestone": str(milestone_path) if milestone_path else None}


def load_recovery_checkpoint(path: Path, *, corpus_sha256: str | None = None) -> dict:
    path = Path(path)
    before = _file_sha256(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    after = _file_sha256(path)
    if before != after:
        raise RuntimeError("cellular recovery checkpoint changed while loading")
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("source_sha256") != source_sha256():
        raise ValueError("cellular recovery checkpoint provenance drifted")
    if corpus_sha256 is not None and payload.get("corpus_sha256") != corpus_sha256:
        raise ValueError("cellular recovery checkpoint corpus drifted")
    if int(payload.get("step", 0)) < 1:
        raise ValueError("cellular recovery checkpoint step drifted")
    return payload
