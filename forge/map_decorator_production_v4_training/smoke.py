from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Final
import uuid

import torch

from ..map_decorator.hashing import json_sha256
from ..map_decorator_ml.contract import ModelConfig
from ..map_decorator_production_v2.training import WarmStartEMA
from ..map_decorator_production_v4.contract import ProposalLocatorConfig
from ..map_decorator_production_v4.model import ProposalConditionedDecoratorV4
from ..map_decorator_production_v4.proposal import ProposalAuthority
from ..safety import require_disk_floor
from .checkpoint import (
    file_sha256,
    inspect_checkpoint,
    load_checkpoint,
    save_checkpoint,
    tensor_state_sha256,
    training_source_sha256,
)
from .contract import ResidualLossConfig, ResidualTrainingConfig, V4_TRAINING_CONTRACT_SHA256
from .dataset import ProposalTeacherSample, collate_proposal_samples
from .training import make_optimizer, train_batch


SMOKE_FORMAT: Final[str] = "nullvector-map-decorator-v4-residual-training-smoke/1.0.0"
REPORT_NAME: Final[str] = "smoke_report.json"
CHECKPOINT_NAME: Final[str] = "checkpoint_step_0001.pt"
CORE = ModelConfig(base_channels=4, condition_channels=8)
LOCATOR = ProposalLocatorConfig(locator_channels=4, locator_blocks=1, count_hidden_channels=4)
TRAINING = ResidualTrainingConfig(ema_decay=0.9, full_mask_stride=1)
LOSS = ResidualLossConfig()
SEED: Final[int] = 0x44D3C011


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fresh():
    torch.manual_seed(SEED)
    model = ProposalConditionedDecoratorV4(CORE, LOCATOR)
    optimizer = make_optimizer(model, TRAINING)
    ema = WarmStartEMA(model, TRAINING.ema_decay)
    generator = torch.Generator().manual_seed(SEED)
    return model, optimizer, ema, generator


def _step(model, optimizer, ema, generator, batch):
    return train_batch(
        model,
        optimizer,
        ema,
        batch,
        generator=generator,
        training_config=TRAINING,
        loss_config=LOSS,
    )


def _authoritative_batch(authority: ProposalAuthority):
    sample, proposals = authority.sample_and_proposals(authority.authority.corpus.refs_by_split["test"][0])
    batch = collate_proposal_samples([ProposalTeacherSample(sample, proposals)])
    return sample, proposals, batch


def _replay_two_steps(batch):
    model, optimizer, ema, generator = _fresh()
    initial_sha = tensor_state_sha256(model.state_dict())
    history = [_step(model, optimizer, ema, generator, batch) for _ in range(2)]
    return {
        "initial_model_sha256": initial_sha,
        "history": history,
        "final_model_sha256": tensor_state_sha256(model.state_dict()),
        "final_ema_sha256": tensor_state_sha256(ema.shadow),
        "ema_updates": ema.updates,
    }


def run_smoke(corpus_root: Path, index_root: Path, output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("V4 residual smoke output is immutable.")
    require_disk_floor(output.parent, floor_gb=100.0, planned_bytes=256 * 1024 * 1024)
    if torch.cuda.is_initialized():
        raise RuntimeError("V4 residual CPU smoke refuses CUDA initialization.")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    sample, proposals, batch = _authoritative_batch(authority)

    replay = _replay_two_steps(batch)
    initial_sha = replay["initial_model_sha256"]
    full_history = replay["history"]
    full_model_sha = replay["final_model_sha256"]
    full_ema_sha = replay["final_ema_sha256"]

    resumed_model, resumed_optimizer, resumed_ema, resumed_generator = _fresh()
    first = _step(resumed_model, resumed_optimizer, resumed_ema, resumed_generator, batch)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    checkpoint_path = staging / CHECKPOINT_NAME
    sidecar = save_checkpoint(
        checkpoint_path,
        resumed_model,
        resumed_optimizer,
        resumed_ema,
        resumed_generator,
        core_config=CORE,
        locator_config=LOCATOR,
        training_config=TRAINING,
        loss_config=LOSS,
        global_step=1,
        corpus_sha256=authority.authority.corpus.corpus_sha256,
        index_semantic_sha256=authority.authority.index_semantic_sha256,
        metrics={"first_step": first},
    )
    reload_model, reload_optimizer, reload_ema, reload_generator = _fresh()
    loaded = load_checkpoint(
        checkpoint_path,
        reload_model,
        reload_optimizer,
        reload_ema,
        reload_generator,
        expected_step=1,
        expected_corpus_sha256=authority.authority.corpus.corpus_sha256,
        expected_index_semantic_sha256=authority.authority.index_semantic_sha256,
        expected_training_config=TRAINING,
        expected_loss_config=LOSS,
    )
    second = _step(reload_model, reload_optimizer, reload_ema, reload_generator, batch)
    resumed_model_sha = tensor_state_sha256(reload_model.state_dict())
    resumed_ema_sha = tensor_state_sha256(reload_ema.shadow)
    exact_resume = (
        resumed_model_sha == full_model_sha
        and resumed_ema_sha == full_ema_sha
        and full_history == [first, second]
        and loaded["metrics"] == {"first_step": first}
    )
    report: dict[str, object] = {
        "format": SMOKE_FORMAT,
        "status": "passed",
        "training_contract_sha256": V4_TRAINING_CONTRACT_SHA256,
        "source_sha256": training_source_sha256(),
        "authority": {
            "corpus_sha256": authority.authority.corpus.corpus_sha256,
            "index_semantic_sha256": authority.authority.index_semantic_sha256,
            "sample_identity_sha256": sample.sample_identity_sha256,
            "proposal_fields_sha256": proposals.fields_sha256,
        },
        "config": {
            "core": CORE.to_dict(),
            "locator": LOCATOR.to_dict(),
            "training": TRAINING.to_dict(),
            "loss": LOSS.to_dict(),
            "steps": 2,
        },
        "runtime": {"device": "cpu", "cuda_initialized": torch.cuda.is_initialized(), "threads": torch.get_num_threads()},
        "history": full_history,
        "initial_model_sha256": initial_sha,
        "final_model_sha256": full_model_sha,
        "final_ema_sha256": full_ema_sha,
        "checkpoint": {
            "path": CHECKPOINT_NAME,
            "bytes": checkpoint_path.stat().st_size,
            "sha256": file_sha256(checkpoint_path),
            "sidecar_sha256": sidecar["sidecar_sha256"],
        },
        "gates": {
            "finite_losses": all(bool(torch.isfinite(torch.tensor(item["loss"]["total"]))) for item in full_history),
            "model_updated": initial_sha != full_model_sha,
            "ema_updates_exact": replay["ema_updates"] == 2,
            "checkpoint_inspected": inspect_checkpoint(checkpoint_path)["global_step"] == 1,
            "interrupted_resume_exact": exact_resume,
            "cpu_only": not torch.cuda.is_initialized(),
            "not_a_quality_claim": True,
        },
    }
    if not all(report["gates"].values()):  # type: ignore[union-attr]
        raise RuntimeError(f"V4 residual smoke failed: {report['gates']}")
    report["report_sha256"] = json_sha256(report)
    _atomic_json(staging / REPORT_NAME, report)
    os.replace(staging, output)
    return validate_smoke(output, corpus_root=corpus_root, index_root=index_root)


def validate_smoke(output: Path, *, corpus_root: Path, index_root: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    report_path = output / REPORT_NAME
    if not report_path.is_file() or report_path.is_symlink() or not 0 < report_path.stat().st_size <= 4 * 1024 * 1024:
        raise ValueError("V4 residual smoke report is missing, unsafe, or oversized.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    stored = report.pop("report_sha256", None)
    if stored != json_sha256(report):
        raise ValueError("V4 residual smoke report self-hash failed.")
    report["report_sha256"] = stored
    if report.get("format") != SMOKE_FORMAT or report.get("status") != "passed" or report.get("source_sha256") != training_source_sha256():
        raise ValueError("V4 residual smoke format/source/status failed.")
    if report.get("training_contract_sha256") != V4_TRAINING_CONTRACT_SHA256:
        raise ValueError("V4 residual smoke training contract drifted.")
    if report.get("config") != {
        "core": CORE.to_dict(),
        "locator": LOCATOR.to_dict(),
        "training": TRAINING.to_dict(),
        "loss": LOSS.to_dict(),
        "steps": 2,
    }:
        raise ValueError("V4 residual smoke configuration drifted.")
    if report.get("checkpoint", {}).get("path") != CHECKPOINT_NAME:
        raise ValueError("V4 residual smoke checkpoint path drifted.")
    checkpoint_path = output / CHECKPOINT_NAME
    payload = inspect_checkpoint(checkpoint_path)
    if payload["corpus_sha256"] != report["authority"]["corpus_sha256"] or payload["index_semantic_sha256"] != report["authority"]["index_semantic_sha256"]:
        raise ValueError("V4 residual smoke checkpoint authority drifted.")
    if not all(report["gates"].values()):
        raise ValueError("V4 residual smoke recorded a failed gate.")
    authority = ProposalAuthority.load(Path(corpus_root), Path(index_root))
    sample, proposals, batch = _authoritative_batch(authority)
    expected_authority = {
        "corpus_sha256": authority.authority.corpus.corpus_sha256,
        "index_semantic_sha256": authority.authority.index_semantic_sha256,
        "sample_identity_sha256": sample.sample_identity_sha256,
        "proposal_fields_sha256": proposals.fields_sha256,
    }
    if report.get("authority") != expected_authority:
        raise ValueError("V4 residual smoke authority semantic replay failed.")
    replay = _replay_two_steps(batch)
    expected_replay = {
        "history": replay["history"],
        "initial_model_sha256": replay["initial_model_sha256"],
        "final_model_sha256": replay["final_model_sha256"],
        "final_ema_sha256": replay["final_ema_sha256"],
    }
    for key, value in expected_replay.items():
        if report.get(key) != value:
            raise ValueError(f"V4 residual smoke {key} semantic replay failed.")
    one_model, one_optimizer, one_ema, one_generator = _fresh()
    first = _step(one_model, one_optimizer, one_ema, one_generator, batch)
    if payload.get("model_tensor_sha256") != tensor_state_sha256(one_model.state_dict()):
        raise ValueError("V4 residual smoke checkpoint model semantic replay failed.")
    if payload.get("ema_tensor_sha256") != tensor_state_sha256(one_ema.shadow):
        raise ValueError("V4 residual smoke checkpoint EMA semantic replay failed.")
    if payload.get("metrics") != {"first_step": first}:
        raise ValueError("V4 residual smoke checkpoint metric semantic replay failed.")
    return report
