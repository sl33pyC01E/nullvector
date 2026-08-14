from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import Tensor

from ..map_topology_neural_prior.contract import CODEBOOK_SIZE, MASK_TOKEN
from ..map_topology_neural_prior.masking import tensor_sha256


@dataclass(frozen=True, slots=True)
class SeededParallelSample:
    tokens: Tensor
    uncertainty: Tensor
    trace: tuple[dict[str, object], ...]
    seed: int
    temperature: float
    top_k: int


def sample_seeded_parallel(
    model: torch.nn.Module,
    conditions: dict[str, Tensor],
    *,
    sampling_steps: int,
    seed: int,
    temperature: float,
    top_k: int,
) -> SeededParallelSample:
    """Reveal a fully masked map with order-independent seeded top-k Gumbel sampling."""
    if type(seed) is not int or not 0 <= seed < 1 << 63:
        raise ValueError("Sampling seed must be unsigned 63-bit.")
    if type(sampling_steps) is not int or not 2 <= sampling_steps <= 32:
        raise ValueError("sampling_steps must be in [2,32].")
    if not math.isfinite(temperature) or not 0.05 <= temperature <= 4.0:
        raise ValueError("temperature must be finite and in [0.05,4].")
    if type(top_k) is not int or not 1 <= top_k <= CODEBOOK_SIZE:
        raise ValueError("top_k must be in [1,512].")
    valid = conditions.get("valid_mask")
    if not isinstance(valid, Tensor) or valid.dtype != torch.bool or valid.ndim != 4 or valid.shape[0] != 1 or valid.shape[1] != 1:
        raise ValueError("Seeded free generation requires one CPU boolean valid mask.")
    if valid.device.type != "cpu" or not bool(valid.any()):
        raise ValueError("Seeded free generation is CPU-only and requires valid cells.")
    for name, value in conditions.items():
        if not isinstance(value, Tensor) or value.device.type != "cpu":
            raise ValueError(f"Condition {name!r} must be a CPU tensor.")
    valid_cells = valid[:, 0]
    tokens = torch.zeros(valid_cells.shape, dtype=torch.long)
    tokens[valid_cells] = MASK_TOKEN
    uncertainty = torch.ones(valid_cells.shape, dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    trace: list[dict[str, object]] = []
    with torch.inference_mode():
        for iteration in range(sampling_steps):
            remaining = tokens == MASK_TOKEN
            remaining_count = int(remaining.sum())
            if remaining_count == 0:
                break
            fraction = remaining.sum(dim=(1, 2)).float() / valid_cells.sum(dim=(1, 2)).clamp_min(1)
            logits = model({**conditions, "tokens": tokens, "mask_fraction": fraction[:, None]}).float()
            if logits.shape != (1, CODEBOOK_SIZE, tokens.shape[1], tokens.shape[2]) or not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Masked prior emitted malformed or non-finite logits.")
            sorted_logits, sorted_tokens = torch.sort(logits, dim=1, descending=True, stable=True)
            candidate_logits = sorted_logits[:, :top_k]
            candidate_tokens = sorted_tokens[:, :top_k]
            probabilities = torch.softmax(candidate_logits / temperature, dim=1)
            uniform = torch.rand(candidate_logits.shape, generator=generator).clamp_(1.0e-7, 1.0 - 1.0e-7)
            gumbel = -torch.log(-torch.log(uniform))
            choice = torch.argmax(candidate_logits / temperature + gumbel, dim=1, keepdim=True)
            proposed = torch.gather(candidate_tokens, 1, choice).squeeze(1)
            sampled_probability = torch.gather(probabilities, 1, choice).squeeze(1)
            reveal_target = math.ceil(int(valid_cells.sum()) * (iteration + 1) / sampling_steps)
            revealed = int(valid_cells.sum()) - remaining_count
            reveal_count = max(1, min(remaining_count, reveal_target - revealed))
            candidates = torch.nonzero(remaining.flatten(), as_tuple=False).flatten()
            scores = sampled_probability.flatten().index_select(0, candidates)
            chosen_order = torch.argsort(scores, descending=True, stable=True)[:reveal_count]
            chosen_cells = candidates.index_select(0, chosen_order)
            flat_tokens = tokens.flatten()
            flat_proposed = proposed.flatten()
            flat_uncertainty = uncertainty.flatten()
            flat_probability = sampled_probability.flatten()
            flat_tokens[chosen_cells] = flat_proposed[chosen_cells]
            flat_uncertainty[chosen_cells] = 1.0 - flat_probability[chosen_cells]
            trace.append({
                "iteration": iteration,
                "remaining_before": remaining_count,
                "revealed": reveal_count,
                "tokens_sha256": tensor_sha256(tokens),
                "uncertainty_sha256": tensor_sha256(uncertainty),
            })
    if bool((tokens[valid_cells] == MASK_TOKEN).any()) or bool((tokens[~valid_cells] != 0).any()):
        raise RuntimeError("Seeded sampler failed full valid-cell revelation or padding discipline.")
    if len(trace) != sampling_steps or sum(int(row["revealed"]) for row in trace) != int(valid_cells.sum()):
        raise RuntimeError("Seeded sampler reveal schedule drifted.")
    return SeededParallelSample(
        tokens=tokens,
        uncertainty=uncertainty,
        trace=tuple(trace),
        seed=seed,
        temperature=float(temperature),
        top_k=top_k,
    )
