from __future__ import annotations


FORMAT = "nullvector-living-body-substrate/1.0.0"
SYSTEMS = (
    "integrity", "neural", "circulation", "respiration", "digestion",
    "senses", "locomotion",
)
FAMILY_NAMES = ("humanoid", "animalian", "plantlike", "anomaly", "machine")
ORGAN_SYSTEM = {
    "brain": "neural", "phase_brain": "neural", "processor": "neural", "meristem": "neural",
    "heart": "circulation", "vascular": "circulation", "coolant_pump": "circulation",
    "lung": "respiration", "radiator": "respiration",
    "gut": "digestion", "transmuter": "digestion", "battery": "digestion", "bulb": "digestion",
    "eye": "senses", "photoreceptor": "senses", "singularity": "senses", "optic": "senses",
}
FLUID_TISSUES = {
    "vascular": (1.0, .80), "respiratory": (.62, .42), "digestive": (.78, .58),
    "neural": (.38, .70), "phase": (.48, .16), "machine": (.55, .24),
}
POLYP_FAMILIES = {2, 3, 4}
