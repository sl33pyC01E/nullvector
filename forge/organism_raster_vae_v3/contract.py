from __future__ import annotations

from dataclasses import asdict, dataclass
import math


FORMAT = "nullvector-structured-organism-raster-vae-v3-calibration/1.0.0"
CHECKPOINT_FORMAT = "nullvector-structured-organism-raster-vae-v3-checkpoint/1.0.0"
FAMILY_NAMES = ("humanoid","animalian","plantlike","anomaly","machine")
INPUT_CHANNELS = 42
TISSUE_CLASSES = 15


@dataclass(frozen=True, slots=True)
class RasterVAEV3Config:
    base_width: int = 128
    mid_width: int = 256
    anatomy_width: int = 384
    global_width: int = 640
    depth: int = 3
    global_depth: int = 4
    fine_channels: int = 20
    anatomy_channels: int = 32
    global_channels: int = 48
    condition_dim: int = 256
    beta_fine: float = 1e-4
    beta_anatomy: float = 2e-4
    beta_global: float = 5e-4
    free_bits: float = .015

    def __post_init__(self) -> None:
        ints=(self.base_width,self.mid_width,self.anatomy_width,self.global_width,self.depth,self.global_depth,self.fine_channels,self.anatomy_channels,self.global_channels,self.condition_dim)
        if any(type(value) is not int for value in ints): raise ValueError("VAE v3 integer contract drifted")
        if self.base_width<64 or self.global_width<self.anatomy_width or not 2<=self.depth<=6 or not 2<=self.global_depth<=8: raise ValueError("VAE v3 geometry drifted")
        for value in (self.beta_fine,self.beta_anatomy,self.beta_global,self.free_bits):
            if not math.isfinite(value) or value<0: raise ValueError("VAE v3 regularization drifted")

    def to_dict(self) -> dict[str,int|float]: return asdict(self)
