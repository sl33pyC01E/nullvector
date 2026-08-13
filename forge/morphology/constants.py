from __future__ import annotations

CANVAS_SIZE = 48
SAFETY_MARGIN = 3

FAMILIES = (
    "humanoid",
    "animalian",
    "plantlike",
    "anomaly",
    "machine",
)

LAYER_NAMES = (
    "body",
    "armor",
    "head",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "appendage",
    "weapon",
    "core",
    "detail",
    "emission",
)

(
    BODY,
    ARMOR,
    HEAD,
    LEFT_ARM,
    RIGHT_ARM,
    LEFT_LEG,
    RIGHT_LEG,
    APPENDAGE,
    WEAPON,
    CORE,
    DETAIL,
    EMISSION,
) = range(len(LAYER_NAMES))

STRUCTURAL_LAYERS = (
    BODY,
    ARMOR,
    HEAD,
    LEFT_ARM,
    RIGHT_ARM,
    LEFT_LEG,
    RIGHT_LEG,
    APPENDAGE,
    WEAPON,
    CORE,
)

GENOME_VERSION = "broad-morphology-genome-v1"
RENDERER_VERSION = "broad-morphology-grammar-v2-role-conditioned"
SEMANTIC_FORMAT = "broad-morphology-semantic-v1"
MANIFEST_FORMAT = "neural-morphology-manifest-v1"

GUIDE_CHANNEL_NAMES = (
    "silhouette",
    "body",
    "skeleton",
    "joints",
    "sockets",
    "core",
    "horizontal_position",
    "root_distance",
)

PART_OWNER_NAMES = (
    "background",
    "body",
    "armor",
    "head",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "appendage",
    "weapon",
    "core",
    "detail",
    "emission",
    "joint",
    "terminal",
    "ornament",
    "aura",
)

MATERIAL_NAMES = (
    "void",
    "tissue",
    "chitin",
    "bark",
    "alloy",
    "armor",
    "weapon",
    "organ",
    "marking",
    "energy",
)

EMISSION_LEVEL_NAMES = ("off", "reactive", "emissive", "radiant")
ROLE_NAMES = (
    "striker",
    "defender",
    "scout",
    "controller",
    "support",
    "artillery",
    "harvester",
    "disruptor",
)
SUBTYPE_NAMES = tuple(
    f"{family}_{variant}"
    for family in FAMILIES
    for variant in range(4)
)

JOINT_LAYER = {
    "root": BODY,
    "head": HEAD,
    "left_shoulder": LEFT_ARM,
    "right_shoulder": RIGHT_ARM,
    "left_hip": LEFT_LEG,
    "right_hip": RIGHT_LEG,
    "appendage_base": APPENDAGE,
    "weapon_mount": WEAPON,
}

SOCKET_LAYER = {
    "focus": CORE,
    "muzzle": WEAPON,
    "left_hand": LEFT_ARM,
    "right_hand": RIGHT_ARM,
    "left_foot": LEFT_LEG,
    "right_foot": RIGHT_LEG,
    "appendage_tip": APPENDAGE,
}
