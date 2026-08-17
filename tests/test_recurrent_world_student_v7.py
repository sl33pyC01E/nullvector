from __future__ import annotations
import inspect
from forge.recurrent_world_student_v7 import contract,evaluation,training

def test_v7_binds_calibrated_v6_and_adapted_decoder():
    assert contract.PARENT_SHA256=="1516633d413aa19930dea53d0eb5a526d8528761e4120f4a0e9b70da42489b64"
    assert contract.CODEC_SHA256=="8b29795559876ce9e067e3b7a1addd72911d0704af43612afd887e227f86947a"
def test_v7_backpropagates_decoder_pixel_and_edge_losses():
    source=inspect.getsource(training.train)
    assert "codec.model.requires_grad_(False)" in source
    assert "codec.model.decode(next_latent" in source
    assert "codec.model.decode(target[:pixel_count])" in source
    assert "visual_loss.backward()" in source
    assert "pixel_weight" in source and "edge_weight" in source
    assert "anchor.gated_action" in source
    assert "parent_anchor_weight" in source
    assert "visual_update=update%plan.pixel_every==0" in source
    assert contract.TrainingPlan().pixel_batch_size==2
    assert contract.TrainingPlan().pixel_every==32
    assert contract.TrainingPlan().batch_size==64
    assert "latent_edge_weight" in source and "latent_moment_weight" in source
def test_v7_selection_and_test_include_pixel_space_gates():
    source=inspect.getsource(evaluation.evaluate)
    assert "min(rows" in source and "sequences[5]" in source
    assert "all_pixel_horizons_beat_persistence" in source
    assert "pixel_motion_at_32" in source
    pixel_source=inspect.getsource(training._pixel_metrics)
    assert "codec.model.decode(initial)" in pixel_source
