from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

from ..map_topology_neural_prior.contract import MaskedPriorConfig
from ..map_topology_neural_prior.masking import MASK_MODES


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
CALIBRATION_FORMAT: Final[str] = "nullvector-neural-map-topology-prior-calibration/1.0.0"
CHECKPOINT_FORMAT: Final[str] = "nullvector-neural-map-topology-prior-production-checkpoint/1.0.0"
FROZEN_LATENT_CORPUS_RELATIVE: Final[str] = "outputs/map_topology_neural_prior_corpus/v1"
FROZEN_LATENT_CORPUS_MANIFEST_FILE_SHA256: Final[str] = "12ae282fe1d89f4b8f5c87d0d5acf1a8eddf7ab15cd2d32031a6bf7ba1cc3b96"
FROZEN_LATENT_CORPUS_MANIFEST_SHA256: Final[str] = "01df481c1b3300e41c0e9a70153679e48a9483fd30b0ac3b4e800cff3d198359"
FROZEN_LATENT_CORPUS_IDENTITY_SHA256: Final[str] = "bbcce0606f12d04d53e15e50c16852a8ee3d0e7262146e4c85c5965cf10f4d56"
FROZEN_LATENT_CORPUS_SOURCE_SHA256: Final[str] = "bf321bdb745cbf70107ef6f0390b6c1d86339180935f1bd88e0961084008c2c8"
FROZEN_PRIOR_SOURCE_SHA256: Final[str] = "76fcbce48e1ce20f5e1f28c20a38cc9c9d8c98be2cedccd221e7f95bb6145e15"
SOURCE_FILES: Final[tuple[str, ...]] = (
    "forge/map_topology_neural_prior_training/__init__.py",
    "forge/map_topology_neural_prior_training/__main__.py",
    "forge/map_topology_neural_prior_training/checkpoint.py",
    "forge/map_topology_neural_prior_training/contract.py",
    "forge/map_topology_neural_prior_training/dataset.py",
    "forge/map_topology_neural_prior_training/metrics.py",
    "forge/map_topology_neural_prior_training/training.py",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(root: Path = PROJECT_ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_FILES:
        path = Path(root) / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def training_source_sha256(root: Path = PROJECT_ROOT) -> str:
    return hashlib.sha256(canonical_json_bytes(source_manifest(root))).hexdigest()


@dataclass(frozen=True, slots=True)
class PriorCalibrationConfig:
    steps: int = 200
    width: int = 64
    residual_depth: int = 4
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip: float = 1.0
    model_ema_decay: float = 0.995
    cell_budget: int = 32_768
    maximum_batch_size: int = 16
    minimum_mask_fraction: float = 0.15
    maximum_mask_fraction: float = 0.95
    validation_samples: int = 48
    test_samples: int = 24
    seed: int = 0x5052494F5243414C

    def __post_init__(self) -> None:
        if type(self.steps) is not int or not 1 <= self.steps <= 500:
            raise ValueError("Prior calibration steps must be in [1,500].")
        if not 16 <= self.width <= 128 or not 1 <= self.residual_depth <= 6:
            raise ValueError("Prior calibration model dimensions are invalid.")
        floats = (self.learning_rate, self.weight_decay, self.gradient_clip, self.model_ema_decay, self.minimum_mask_fraction, self.maximum_mask_fraction)
        if any(not math.isfinite(value) for value in floats):
            raise ValueError("Prior calibration floating configuration must be finite.")
        if not 0 < self.learning_rate <= 0.01 or not 0 <= self.weight_decay <= 1 or not 0 < self.gradient_clip <= 100 or not 0 <= self.model_ema_decay < 1:
            raise ValueError("Prior calibration optimizer configuration is invalid.")
        if not 0 < self.minimum_mask_fraction < self.maximum_mask_fraction <= 1:
            raise ValueError("Prior calibration mask bounds are invalid.")
        if not 4_096 <= self.cell_budget <= 262_144 or not 1 <= self.maximum_batch_size <= 64:
            raise ValueError("Prior calibration batch bounds are invalid.")
        if self.validation_samples not in {6, 48} or self.test_samples not in {6, 24}:
            raise ValueError("Prior calibration evaluation census is invalid.")
        if type(self.seed) is not int or not 0 <= self.seed < 1 << 63:
            raise ValueError("Prior calibration seed must be unsigned 63-bit.")

    def model_config(self) -> MaskedPriorConfig:
        return MaskedPriorConfig(
            width=self.width, residual_depth=self.residual_depth, steps=2,
            learning_rate=self.learning_rate, weight_decay=self.weight_decay,
            gradient_clip=self.gradient_clip, model_ema_decay=self.model_ema_decay,
            minimum_mask_fraction=self.minimum_mask_fraction,
            maximum_mask_fraction=self.maximum_mask_fraction,
            sampling_steps=8, seed=self.seed,
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model"] = self.model_config().to_dict()
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PriorCalibrationConfig":
        if not isinstance(payload, dict):
            raise ValueError("Prior calibration config must be a mapping.")
        values = dict(payload)
        model = values.pop("model", None)
        if set(values) != set(asdict(cls())):
            raise ValueError("Prior calibration config members drifted.")
        config = cls(**values)
        if model != config.model_config().to_dict():
            raise ValueError("Prior calibration model metadata drifted.")
        return config


QUALITY_GATES: Final[dict[str, float]] = {
    "minimum_loss_improvement": 0.10,
    "minimum_validation_accuracy": 0.08,
    "minimum_test_accuracy": 0.06,
    "minimum_validation_macro_mode_accuracy": 0.04,
    "minimum_test_macro_mode_accuracy": 0.03,
}

HISTORY_KEYS: Final[set[str]] = {
    "step", "loss", "gradient_norm", "batch_size", "shape", "masked_cells",
    "mask_fraction_mean", "modes", "sample_registry_sha256",
}
EVALUATION_KEYS: Final[set[str]] = {
    "sample_count", "sample_registry_sha256", "masked_cells", "loss",
    "accuracy", "macro_mode_accuracy", "modes", "vocabulary_size",
}
MODE_METRIC_KEYS: Final[set[str]] = {"masked_cells", "accuracy"}
SAFETY_GATE_KEYS: Final[set[str]] = {
    "step_count_exact", "finite_history", "model_updated",
    "evaluation_census_exact", "train_split_only", "latent_corpus_exact",
    "raw_generation_disabled", "compiler_disabled", "godot_integration_disabled",
    "disk_floor_preserved",
}
CLAIM_KEYS: Final[set[str]] = {
    "masked_token_calibration_only", "quality_milestone_reached",
    "raw_generation_published", "compiled_maps_published", "godot_integration",
}
RUNTIME_KEYS: Final[set[str]] = {
    "device", "precision", "training_seconds", "evaluation_seconds",
    "elapsed_seconds", "peak_allocated_bytes", "peak_reserved_bytes",
}


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def validate_history(history: Any, config: PriorCalibrationConfig) -> list[dict[str, Any]]:
    if not isinstance(history, list) or len(history) != config.steps:
        raise ValueError("Prior calibration history census drifted.")
    for expected_step, row in enumerate(history, start=1):
        if not isinstance(row, dict) or set(row) != HISTORY_KEYS or row["step"] != expected_step:
            raise ValueError("Prior calibration history schema drifted.")
        if any(type(row[name]) not in (int, float) or isinstance(row[name], bool) or not math.isfinite(row[name]) for name in ("loss", "gradient_norm", "mask_fraction_mean")):
            raise ValueError("Prior calibration history contains non-finite metrics.")
        if row["loss"] <= 0 or row["gradient_norm"] < 0 or not 0 < row["mask_fraction_mean"] < 1:
            raise ValueError("Prior calibration history metrics are outside their domains.")
        if type(row["batch_size"]) is not int or not 1 <= row["batch_size"] <= config.maximum_batch_size:
            raise ValueError("Prior calibration batch size drifted.")
        if type(row["masked_cells"]) is not int or row["masked_cells"] <= 0:
            raise ValueError("Prior calibration masked-cell census drifted.")
        if not isinstance(row["shape"], list) or len(row["shape"]) != 2 or any(type(value) is not int or value <= 0 for value in row["shape"]):
            raise ValueError("Prior calibration history shape drifted.")
        if not isinstance(row["modes"], list) or len(row["modes"]) != row["batch_size"] or any(mode not in MASK_MODES for mode in row["modes"]):
            raise ValueError("Prior calibration mask-mode history drifted.")
        if not is_sha256(row["sample_registry_sha256"]):
            raise ValueError("Prior calibration sample registry hash is malformed.")
    return history


def validate_metric(metric: Any, expected_samples: int) -> dict[str, Any]:
    if not isinstance(metric, dict) or set(metric) != EVALUATION_KEYS:
        raise ValueError("Prior calibration evaluation schema drifted.")
    if metric["sample_count"] != expected_samples or type(metric["masked_cells"]) is not int or metric["masked_cells"] <= 0:
        raise ValueError("Prior calibration evaluation census drifted.")
    if metric["vocabulary_size"] != 512 or not is_sha256(metric["sample_registry_sha256"]):
        raise ValueError("Prior calibration evaluation authority drifted.")
    for name in ("loss", "accuracy", "macro_mode_accuracy"):
        if type(metric[name]) not in (int, float) or isinstance(metric[name], bool) or not math.isfinite(metric[name]):
            raise ValueError("Prior calibration evaluation contains non-finite metrics.")
    if metric["loss"] <= 0 or not 0 <= metric["accuracy"] <= 1 or not 0 <= metric["macro_mode_accuracy"] <= 1:
        raise ValueError("Prior calibration evaluation metrics are outside their domains.")
    modes = metric["modes"]
    if not isinstance(modes, dict) or set(modes) != set(MASK_MODES):
        raise ValueError("Prior calibration evaluation mask modes drifted.")
    counted = 0
    for mode in MASK_MODES:
        row = modes[mode]
        if not isinstance(row, dict) or set(row) != MODE_METRIC_KEYS or type(row["masked_cells"]) is not int or row["masked_cells"] <= 0:
            raise ValueError("Prior calibration mode metric census drifted.")
        if type(row["accuracy"]) not in (int, float) or isinstance(row["accuracy"], bool) or not math.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1:
            raise ValueError("Prior calibration mode accuracy drifted.")
        counted += row["masked_cells"]
    if counted != metric["masked_cells"]:
        raise ValueError("Prior calibration mode totals disagree.")
    expected_macro = sum(modes[mode]["accuracy"] for mode in MASK_MODES) / len(MASK_MODES)
    if metric["macro_mode_accuracy"] != expected_macro:
        raise ValueError("Prior calibration macro mask-mode metric drifted.")
    return metric


def validate_evaluation(evaluation: Any, config: PriorCalibrationConfig) -> dict[str, Any]:
    if not isinstance(evaluation, dict) or set(evaluation) != {"validation", "test"}:
        raise ValueError("Prior calibration evaluation split census drifted.")
    for split, expected in (("validation", config.validation_samples), ("test", config.test_samples)):
        group = evaluation[split]
        if not isinstance(group, dict) or set(group) != {"baseline", "raw", "ema"}:
            raise ValueError("Prior calibration evaluation model census drifted.")
        registry = None
        for mode in ("baseline", "raw", "ema"):
            metric = validate_metric(group[mode], expected)
            registry = metric["sample_registry_sha256"] if registry is None else registry
            if metric["sample_registry_sha256"] != registry:
                raise ValueError("Prior calibration evaluation samples disagree between models.")
    return evaluation
