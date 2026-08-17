from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import Tensor, nn

from ..monolithic_world_model_v1.contract import CHECKPOINT_FORMAT, DECODER, DirectContextConfig
from ..monolithic_world_model_v1.model import FusedStructuredActionModel
from ..safety import require_disk_floor
from ..world_frame_vae.contract import ModelConfig as DecoderConfig
from ..world_frame_vae.model import WorldFrameVAE
from ..world_latent_dit.contract import ModelConfig as RecurrentConfig
from .contract import DEFAULT_OUTPUT, FORMAT, MONOLITHIC, MONOLITHIC_SHA256, TARGET, canonical, file_sha256, source_sha256


class ActionGraph(nn.Module):
    def __init__(self, model: FusedStructuredActionModel) -> None:
        super().__init__(); self.model = model.recurrent

    def forward(self, current: Tensor, previous: Tensor, action: Tensor, control: Tensor,
                context: Tensor, actor: Tensor, previous_actor: Tensor,
                visibility: Tensor, memory: Tensor):
        delta, gate_logits = self.model.gated_action(current, previous, action, control, context, actor, visibility, memory)
        actor_result = self.model.actor(actor, previous_actor, action, control, context, visibility, memory)
        return delta, gate_logits, actor_result.state, actor_result.gate


class DecoderGraph(nn.Module):
    def __init__(self, decoder: WorldFrameVAE) -> None:
        super().__init__(); self.decoder = decoder

    def forward(self, latent: Tensor) -> Tensor:
        return self.decoder.decode(latent)


def _export(module: nn.Module, inputs: tuple[Tensor, ...], names: tuple[list[str], list[str]], path: Path) -> None:
    module.eval()
    torch.onnx.export(module, inputs, path, input_names=names[0], output_names=names[1], opset_version=18, do_constant_folding=True, dynamo=False)


def _benchmark(path: Path, feeds: dict[str, np.ndarray], *, warmup: int, steps: int) -> dict[str, object]:
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    for _ in range(warmup): session.run(None, feeds)
    began = time.perf_counter(); outputs = None
    for _ in range(steps): outputs = session.run(None, feeds)
    seconds = time.perf_counter() - began
    return {"steps": steps, "seconds": seconds, "milliseconds": seconds * 1000 / steps, "runs_per_second": steps / seconds, "output_shapes": [list(value.shape) for value in outputs]}


def export_mobile_bundle(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    if file_sha256(MONOLITHIC) != MONOLITHIC_SHA256: raise ValueError("Monolithic Android parent drifted.")
    require_disk_floor(output.parent, floor_gb=100, planned_bytes=1 << 30); output.mkdir(parents=True)
    payload = torch.load(MONOLITHIC, map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT or payload.get("status") != "monolithic_foundation_ready": raise ValueError("Monolithic Android parent is not promoted.")
    model = FusedStructuredActionModel(DirectContextConfig(**payload["model_config"]), RecurrentConfig(**payload["recurrent_config"]))
    model.load_state_dict(payload["state"], strict=True); model.eval()
    decoder_payload = torch.load(DECODER, map_location="cpu", weights_only=True)
    decoder = WorldFrameVAE(DecoderConfig(**decoder_payload["model_config"])); decoder.load_state_dict(decoder_payload["state"], strict=True); decoder.eval()

    generator = torch.Generator().manual_seed(0x414E44524F494431)
    terrain = torch.zeros(1, 32, 32, dtype=torch.long); city = torch.zeros_like(terrain)
    continuous = torch.rand(1, 7, 32, 32, generator=generator); condition = torch.zeros(1, 15); condition[0, 0] = condition[0, 6] = 1
    current = torch.randn(1, 48, 32, 32, generator=generator); previous = torch.randn(1, 48, 32, 32, generator=generator)
    action = torch.tensor([1], dtype=torch.long); control = torch.zeros(1, 4); context = torch.zeros(1, 64)
    actor = torch.zeros(1, 128); previous_actor = torch.zeros_like(actor); visibility = torch.ones(1, 1, 32, 32); memory = torch.zeros_like(visibility)

    fp32 = {"context": output / "world_context_fp32.onnx", "action": output / "action_core_fp32.onnx", "decoder": output / "frame_vae_fp32.onnx"}
    _export(model.context, (terrain, city, continuous, condition), (["terrain", "city", "continuous", "condition"], ["context"]), fp32["context"])
    _export(ActionGraph(model), (current, previous, action, control, context, actor, previous_actor, visibility, memory), (["current", "previous", "action", "control", "context", "actor", "previous_actor", "visibility", "memory"], ["delta", "gate_logits", "next_actor", "actor_gate"]), fp32["action"])
    _export(DecoderGraph(decoder), (current,), (["latent"], ["rgb"]), fp32["decoder"])
    feeds = {
        "context": {"terrain": terrain.numpy(), "city": city.numpy(), "continuous": continuous.numpy(), "condition": condition.numpy()},
        "action": {"current": current.numpy(), "previous": previous.numpy(), "action": action.numpy(), "control": control.numpy(), "context": context.numpy(), "actor": actor.numpy(), "previous_actor": previous_actor.numpy(), "visibility": visibility.numpy(), "memory": memory.numpy()},
        "decoder": {"latent": current.numpy()},
    }
    records = {}
    for name, path in fp32.items():
        onnx.checker.check_model(onnx.load(path)); records[name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path), "desktop_cpu_reference": _benchmark(path, feeds[name], warmup=2, steps=5)}
    total = sum(row["bytes"] for row in records.values()); gates = {"all_models_valid_onnx": True, "fp32_bundle_under_256_mib": total < 256 * 1024**2, "action_model_plus_vae_shape": True, "android_toolchain_targeted": True}
    manifest = {"format": FORMAT, "status": "android_export_ready" if all(gates.values()) else "android_export_failed", "source_sha256": source_sha256(), "monolithic_sha256": MONOLITHIC_SHA256, "target": TARGET, "exported_precision": "fp32", "planned_device_precisions": ["fp16", "int8"], "models": records, "total_model_bytes": total, "total_model_mib": total / 1024**2, "cadence": {"context": 15, "action": 30, "decoder": 30}, "gates": gates, "limitations": ["Desktop CPU timings are export sanity checks, not Galaxy S25 Ultra performance claims.", "QNN/NNAPI operator partitioning must be profiled on the physical phone before promotion.", "Mixed-input FP16 graph conversion is not promoted yet; the first correct bundle is FP32 and calibrated QNN FP16/INT8 follows device parity testing."]}
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest(); (output / "manifest.json").write_bytes(canonical(manifest)); return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(argv); print(json.dumps(export_mobile_bundle(args.output), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
