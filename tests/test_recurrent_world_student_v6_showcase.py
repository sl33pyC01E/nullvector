from __future__ import annotations
import inspect
from forge.recurrent_world_student_v6 import showcase


def test_showcase_is_continuous_recurrent_decode_not_independent_samples():
    source=inspect.getsource(showcase._rollout)
    assert "previous,current=current,next_latent" in source
    assert "logits+applied_bias" in source
    assert "rows.append(current.float().cpu())" in source


def test_showcase_binds_checkpoint_corpus_codec_and_ffmpeg():
    source=inspect.getsource(showcase.build)
    for token in ("CODEC_SHA256","runtime_sha","manifest_sha256","ffmpeg","frame_tree_sha256"):
        assert token in source
