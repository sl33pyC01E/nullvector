from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
import torch.nn.functional as F

from ..config import PROJECT_ROOT
from ..whole_viewport_latent_v1.contract import ModelConfig, canonical
from ..whole_viewport_latent_v1.decoder import load_decoder
from ..whole_viewport_latent_v1.model import WholeViewportLatentModel


DEFAULT_ACTION_RELEASE = PROJECT_ROOT / "outputs/whole_viewport_latent_v1/production_macro_adapted_vae_v4"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/android_viewport_action_v1/production_v1"


class EncoderGraph(nn.Module):
    def __init__(self, vae):
        super().__init__(); self.vae = vae

    def forward(self, rgb):
        return self.vae.encode(rgb)[0]


class ActionGraph(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, previous_latent, spatial, organisms, organism_mask, state, actor_state, actor_field, visibility, memory, control, action):
        return self.model(previous_latent, spatial, organisms, organism_mask, state, actor_state, actor_field, visibility, memory, control, action)


class MobileGpuActionGraph(nn.Module):
    """GPU-safe action graph with numeric fields packed before inference.

    Raw organism tokens are averaged into their 32x32 cells by FoundationWorld.
    The trained token MLP is then applied pointwise. This removes ScatterElements,
    bool casts, and integer embedding lookup from the mobile GPU graph.
    """
    def __init__(self, model):
        super().__init__(); self.model = model

    def forward(self, previous_latent, spatial, organism_field, state, actor_state, actor_field, visibility, memory, control, action_one_hot):
        scene = torch.cat((previous_latent, spatial, actor_field, visibility, memory), 1)
        hidden = self.model.scene(scene)
        tokens = organism_field.permute(0, 2, 3, 1)
        embedded = self.model.organism(tokens).permute(0, 3, 1, 2)
        hidden = hidden + self.model.organism_mix(embedded)
        global_state = torch.cat((state, actor_state, control), 1)
        condition = self.model.global_condition(global_state) + F.linear(action_one_hot, self.model.action.weight.T)
        for block in self.model.blocks: hidden = block(hidden, condition)
        return previous_latent + torch.sigmoid(self.model.gate(hidden)) * self.model.out(hidden)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_action(release: Path):
    release = Path(release).resolve(); raw = (release / "manifest.json").read_bytes(); manifest = json.loads(raw)
    if raw != canonical(manifest): raise ValueError("viewport action manifest is not canonical")
    artifact = release / manifest["artifact"]["path"]
    if artifact.stat().st_size != manifest["artifact"]["bytes"] or _sha(artifact) != manifest["artifact"]["sha256"]: raise ValueError("viewport action artifact drifted")
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    config = ModelConfig(**payload["model_config"]); model = WholeViewportLatentModel(config)
    model.load_state_dict(payload["state"]); model.eval()
    return model, config, manifest


def export(*, action_release: Path = DEFAULT_ACTION_RELEASE, output: Path = DEFAULT_OUTPUT, install_assets: bool = False, mobile_gpu: bool = False):
    output = Path(output); action_release = Path(action_release)
    if output.exists(): raise FileExistsError(output)
    model, config, action_manifest = _load_action(action_release)
    vae, vae_provenance = load_decoder(torch.device("cpu"), Path(action_manifest["decoder"]["release"]))
    encoder = EncoderGraph(vae).eval(); action = (MobileGpuActionGraph(model) if mobile_gpu else ActionGraph(model)).eval()
    staging = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"; staging.mkdir(parents=True)
    generator = torch.Generator().manual_seed(0x53544154454C4F43)
    rgb = torch.rand(1, 3, 256, 256, generator=generator)
    if mobile_gpu:
        args = (
            torch.randn(1, 48, 32, 32, generator=generator), torch.randn(1, 68, 32, 32, generator=generator),
            torch.randn(1, 164, 32, 32, generator=generator), torch.randn(1, 64, generator=generator),
            torch.randn(1, 128, generator=generator), torch.randn(1, 8, 32, 32, generator=generator),
            torch.ones(1, 1, 32, 32), torch.ones(1, 1, 32, 32), torch.zeros(1, 4), torch.nn.functional.one_hot(torch.zeros(1, dtype=torch.long), 22).float(),
        )
        names = ["previous_latent", "spatial", "organism_field", "state", "actor_state", "actor_field", "visibility", "memory", "control", "action_one_hot"]
    else:
        args = (
            torch.randn(1, 48, 32, 32, generator=generator), torch.randn(1, 68, 32, 32, generator=generator),
            torch.randn(1, 64, 164, generator=generator), torch.ones(1, 64, dtype=torch.bool),
            torch.randn(1, 64, generator=generator), torch.randn(1, 128, generator=generator),
            torch.randn(1, 8, 32, 32, generator=generator), torch.ones(1, 1, 32, 32), torch.ones(1, 1, 32, 32),
            torch.zeros(1, 4), torch.zeros(1, dtype=torch.long),
        )
        names = ["previous_latent", "spatial", "organisms", "organism_mask", "state", "actor_state", "actor_field", "visibility", "memory", "control", "action"]
    encoder_path = staging / "viewport_encoder_fp32.onnx"
    action_path = staging / "viewport_action_fp32.onnx"
    torch.onnx.export(encoder, (rgb,), encoder_path, input_names=["rgb"], output_names=["latent"], opset_version=18, do_constant_folding=True, dynamo=False)
    torch.onnx.export(action, args, action_path, input_names=names, output_names=["next_latent"], opset_version=18, do_constant_folding=True, dynamo=False)
    options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
    artifacts = {}
    for path in (encoder_path, action_path):
        onnx.checker.check_model(onnx.load(path)); artifacts[path.name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha(path)}
    encoder_session = ort.InferenceSession(str(encoder_path), sess_options=options, providers=["CPUExecutionProvider"])
    action_session = ort.InferenceSession(str(action_path), sess_options=options, providers=["CPUExecutionProvider"])
    expected_encoder = encoder(rgb).detach().numpy(); actual_encoder = encoder_session.run(None, {"rgb": rgb.numpy()})[0]
    expected_action = action(*args).detach().numpy(); feeds = {name: value.numpy() for name, value in zip(names, args)}; actual_action = action_session.run(None, feeds)[0]
    parity = {"encoder_max_abs": float(np.max(np.abs(expected_encoder - actual_encoder))), "action_max_abs": float(np.max(np.abs(expected_action - actual_action)))}
    gates = {"encoder_onnx_valid": True, "action_onnx_valid": True, "encoder_parity": parity["encoder_max_abs"] < 2e-5, "action_parity": parity["action_max_abs"] < 2e-5, "matching_decoder": action_manifest["decoder"] == vae_provenance}
    gates["all_passed"] = all(gates.values())
    manifest = {"format": "nullvector-android-state-aligned-viewport-action/1.0.0", "status": "ready" if gates["all_passed"] else "rejected", "action_release": {"manifest_sha256": action_manifest["manifest_sha256"], "artifact_sha256": action_manifest["artifact"]["sha256"], "validation": action_manifest["validation"], "status": action_manifest["status"]}, "vae": vae_provenance, "model_config": config.__dict__, "artifacts": artifacts, "parity": parity, "gates": gates, "mobile_gpu_graph": mobile_gpu, "runtime_contract": {"authoritative_world": "FoundationWorld", "latent_initialization": "encode current live scaffold frame", "renderer_switch_resets_world": False, "sample_latent_allowed": False, "organism_projection": "raw token cell-average then trained pointwise token MLP" if mobile_gpu else "trained token MLP then scatter"}}
    manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest(); (staging / "manifest.json").write_bytes(canonical(manifest))
    if not gates["all_passed"]: raise RuntimeError("Android viewport action export failed")
    if install_assets:
        asset_root = PROJECT_ROOT / "android/nullvector-mobile/app/src/main/assets"
        for path in (encoder_path, action_path):
            target = asset_root / path.name; temporary = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}"); shutil.copyfile(path, temporary); os.replace(temporary, target)
    os.replace(staging, output); return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--action-release", type=Path, default=DEFAULT_ACTION_RELEASE); parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--install-assets", action="store_true"); parser.add_argument("--mobile-gpu", action="store_true"); args = parser.parse_args(argv)
    print(json.dumps(export(action_release=args.action_release, output=args.output, install_assets=args.install_assets, mobile_gpu=args.mobile_gpu), indent=2))


if __name__ == "__main__": main()
