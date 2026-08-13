from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import uuid

import torch

from ..map_decorator.features import encode_features
from ..maps.io import load_map_pack
from ..safety import require_disk_floor
from .artifacts import validate_prediction_pack, write_prediction_pack
from .checkpoint import save_checkpoint, source_sha256
from .contract import MODEL_CONTRACT_SHA256, ModelConfig
from .dataset import (
    TeacherRecord,
    assert_no_split_leakage,
    collate_teacher_samples,
    corpus_identity,
    load_teacher_sample,
)
from .model import CategoricalRefinementUNet
from .sampling import SamplerConfig, sample_refinement
from .training import EMA, TrainingConfig, train_batch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-only topology-locked decorator smoke run")
    parser.add_argument("--packs", type=Path, default=Path("outputs/maps_topology_v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/map_decorator_ml_smoke"))
    parser.add_argument("--maps", type=int, default=2)
    parser.add_argument("--train-steps", type=int, default=2)
    parser.add_argument("--refinement-steps", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--condition-channels", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0x5A0CE)
    parser.add_argument("--threads", type=int, default=2)
    return parser


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    if not 1 <= args.maps <= 6 or not 1 <= args.train_steps <= 32:
        raise ValueError("Smoke maps must be in [1,6] and train steps in [1,32].")
    if not 1 <= args.threads <= 4:
        raise ValueError("CPU smoke threads must be in [1,4].")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU smoke refuses to run after CUDA has been initialized.")
    torch.set_num_threads(args.threads)
    if torch.get_num_interop_threads() != 1:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # A library caller may already have entered PyTorch's inter-op pool.
            pass
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Smoke output already exists and will not be overwritten: {output}")
    pack_paths = sorted(path for path in Path(args.packs).resolve().iterdir() if path.is_dir())
    if len(pack_paths) < args.maps:
        raise ValueError(f"Requested {args.maps} packs but found {len(pack_paths)}.")
    selected_packs = pack_paths[: args.maps]
    require_disk_floor(output, planned_bytes=512 * 1024 * 1024)
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        samples = [
            load_teacher_sample(TeacherRecord(path, int(args.seed) + index))
            for index, path in enumerate(selected_packs)
        ]
        assert_no_split_leakage(samples)
        corpus_hash = corpus_identity(samples)
        batch = collate_teacher_samples(samples)
        model_config = ModelConfig(
            base_channels=args.base_channels,
            condition_channels=args.condition_channels,
        )
        training_config = TrainingConfig(seed=args.seed, ema_decay=0.95)
        # Seed only the CPU default generator used by module initialization.
        torch.random.default_generator.manual_seed(training_config.seed)
        model = CategoricalRefinementUNet(model_config)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        ema = EMA(model, training_config.ema_decay)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(training_config.seed)
        history: list[dict[str, object]] = []
        for _ in range(args.train_steps):
            history.append(
                train_batch(
                    model,
                    optimizer,
                    ema,
                    batch,
                    generator=generator,
                    config=training_config,
                    device="cpu",
                )
            )
        checkpoint_path = staging / "smoke_checkpoint.pt"
        checkpoint = save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            ema,
            training_config=training_config,
            corpus_sha256=corpus_hash,
            epoch=1,
            global_step=args.train_steps,
            training_generator=generator,
            metrics=history[-1],
        )
        ema.copy_to(model)
        source_data = load_map_pack(selected_packs[0])
        encoded = encode_features(
            source_data,
            protected_backbone=source_data.protected_backbone,
            required_clearance=source_data.required_clearance,
            decoration_forbidden=source_data.decoration_forbidden,
            public_seed=args.seed,
        )
        sampler_config = SamplerConfig(steps=args.refinement_steps, temperature=0.85)
        first = sample_refinement(
            model,
            source_data,
            encoded,
            generation_seed=args.seed + 1000,
            config=sampler_config,
            device="cpu",
        )
        replay = sample_refinement(
            model,
            source_data,
            encoded,
            generation_seed=args.seed + 1000,
            config=sampler_config,
            device="cpu",
        )
        if first.report["field_sha256"] != replay.report["field_sha256"]:
            raise RuntimeError("CPU seeded sampling replay was not byte-identical.")
        prediction_path = write_prediction_pack(
            staging / "predictions",
            first,
            source_data,
            encoded,
            checkpoint_path=checkpoint_path,
            sampler_config=sampler_config,
            source_sha256=str(checkpoint["source_sha256"]),
            corpus_sha256=corpus_hash,
            ema_tensor_sha256=str(checkpoint["ema_tensor_sha256"]),
        )
        prediction_validation = validate_prediction_pack(
            prediction_path,
            data=source_data,
            encoded=encoded,
            checkpoint_path=checkpoint_path,
        )
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU smoke unexpectedly initialized CUDA.")
        summary: dict[str, object] = {
            "passed": True,
            "device": "cpu",
            "cuda_touched": False,
            "cuda_initialized_after": False,
            "map_count": len(samples),
            "map_ids": [sample.map_id for sample in samples],
            "splits": [sample.split for sample in samples],
            "corpus_sha256": corpus_hash,
            "model_contract_sha256": MODEL_CONTRACT_SHA256,
            "source_sha256": source_sha256(),
            "model_config": model_config.to_dict(),
            "training_config": training_config.to_dict(),
            "train_steps": args.train_steps,
            "history": history,
            "checkpoint": checkpoint,
            "prediction_pack": prediction_path.relative_to(staging).as_posix(),
            "prediction_field_sha256": first.report["field_sha256"],
            "replay_field_sha256": replay.report["field_sha256"],
            "byte_identical_replay": True,
            "prediction_validation": prediction_validation,
            "elapsed_seconds": time.perf_counter() - start,
        }
        (staging / "smoke_report.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return summary
    except BaseException:
        # The uniquely named staging tree is retained as diagnostic evidence.
        raise
    finally:
        torch.use_deterministic_algorithms(prior_deterministic, warn_only=prior_warn_only)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_smoke(args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
