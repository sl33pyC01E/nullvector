"""Discrete semantic latent codec for 48px neural sprite construction."""

from .codec import (
    CodecOutput,
    FSQQuantizer,
    SemanticSpriteFSQ,
    SpriteLatentConfig,
    project_legal_tuples,
    sprite_codec_loss,
)

__all__ = (
    "CodecOutput",
    "FSQQuantizer",
    "SemanticSpriteFSQ",
    "SpriteLatentConfig",
    "project_legal_tuples",
    "sprite_codec_loss",
)
