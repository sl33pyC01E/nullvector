from __future__ import annotations

from dataclasses import dataclass


FORMAT = "nullvector-living-body-graph-dynamics/3.0.0"
CHECKPOINT_FORMAT = "nullvector-living-body-graph-dynamics-checkpoint/3.0.0"
FEATURES = 56
SYSTEMS = 7
FEEDING_TARGETS = 9
ACTION_KINDS = ("idle", "impact", "cut", "heal", "feed", "metabolize", "feed_and_metabolize")

# The feature layout is deliberately public.  Runtime adapters and corpus
# validators use these slices instead of relying on unexplained magic offsets.
FAMILY_SLICE = slice(0, 5)
TISSUE_SLICE = slice(5, 20)
SYSTEM_SLICE = slice(20, 26)
POSITION_SLICE = slice(26, 28)
APPENDAGE_INDEX = 28
SIDE_INDEX = 29
CELL_STATE_SLICE = slice(30, 33)  # health, internal fluid, scar
CONNECTED_INDEX = 33
FEEDER_INDEX = 34
DIGESTIVE_INDEX = 35
ACTION_SLICE = slice(36, 43)
LOCAL_ACTION_SLICE = slice(43, 46)  # impact, cut, heal fields
CONTACT_INDEX = 46
FOOD_MASS_INDEX = 47
NUTRIENT_DENSITY_INDEX = 48
FAMILY_NUTRITION_INDEX = 49
RESERVE_INDEX = 50
FULLNESS_INDEX = 51
CONSUMED_INDEX = 52
ENERGY_INDEX = 53
ACTIVITY_INDEX = 54
DELTA_INDEX = 55


@dataclass(frozen=True, slots=True)
class DynamicsConfig:
    width: int = 256
    depth: int = 6
    family_width: int = 32
    dropout: float = .03

    def __post_init__(self) -> None:
        if not 128 <= self.width <= 768 or not 3 <= self.depth <= 12:
            raise ValueError("living dynamics geometry drifted")
        if not 16 <= self.family_width <= 128 or not 0 <= self.dropout <= .25:
            raise ValueError("living dynamics regularization drifted")


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    segment_steps: int = 250
    batch_size: int = 8
    learning_rate: float = 2e-4
    ema_decay: float = .998
    seed: int = 0x4C4956494E474E4E
