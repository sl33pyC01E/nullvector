from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GENERATION_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "production_handoff_v2"
    / "final_best_stratified80_bank_attempt1"
    / "generation_manifest.json"
)
DEFAULT_STYLE_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "multifield_style"
    / "final_best_stratified80_v3"
    / "style_manifest.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "neural_rig_repair_v2"

REPAIR_VERSION = "logical-neural-rig-repair-v2"
PLAN_FORMAT = "nullvector-neural-rig-repair-plan-v1"
BANK_FORMAT = "nullvector-neural-rig-repair-bank-v1"
REST_AUDIT_FORMAT = "nullvector-neural-rig-repair-rest-audit-v1"
STRESS_SHARD_FORMAT = "nullvector-neural-rig-repair-stress-shard-v1"
STRESS_FORMAT = "nullvector-neural-rig-repair-stress-v1"
REPLAY_FORMAT = "nullvector-neural-rig-repair-replay-v1"

PLAN_SCHEMA = "neural_rig_repair_plan.schema.json"
BANK_SCHEMA = "neural_rig_repair_bank.schema.json"
REPLAY_SCHEMA = "neural_rig_repair_replay.schema.json"

EXPECTED_BRIDGE_SOURCE_SHA256 = (
    "46372e031c91d0202d0e55a8422385978c5157f76d83ed20adef9ed3e7250305"
)
EXPECTED_SAMPLE_COUNT = 80
EXPECTED_BINDABLE_V1 = 70
EXPECTED_REJECTED_V1 = 10
EXPECTED_CLIPS_PER_SAMPLE = 104
EXPECTED_FRAMES_PER_SAMPLE = 944
EXPECTED_CLIP_COUNT = EXPECTED_SAMPLE_COUNT * EXPECTED_CLIPS_PER_SAMPLE
EXPECTED_FRAME_COUNT = EXPECTED_SAMPLE_COUNT * EXPECTED_FRAMES_PER_SAMPLE

EXPECTED_REJECTION_CATEGORIES = {
    "anchor_on_background": 3,
    "plant_topology": 1,
    "required_owner_absence": 3,
    "safety_margin": 3,
}

EXPECTED_REJECTIONS = (
    ("humanoid", 12, "0012_f0_s02_r6_v00", "anchor_on_background"),
    ("plantlike", 8, "0040_f2_s08_r4_v00", "anchor_on_background"),
    ("plantlike", 12, "0044_f2_s10_r6_v00", "anchor_on_background"),
    ("plantlike", 15, "0047_f2_s11_r7_v01", "plant_topology"),
    ("anomaly", 7, "0055_f3_s15_r3_v01", "required_owner_absence"),
    ("anomaly", 8, "0056_f3_s12_r4_v00", "required_owner_absence"),
    ("anomaly", 9, "0057_f3_s12_r4_v01", "required_owner_absence"),
    ("machine", 3, "0067_f4_s17_r1_v01", "safety_margin"),
    ("machine", 8, "0072_f4_s16_r4_v00", "safety_margin"),
    ("machine", 15, "0079_f4_s19_r7_v01", "safety_margin"),
)

REQUIRED_OWNER_IDS = {"body": 1, "head": 3, "core": 10}
AURA_OWNER_ID = 16
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_RAW_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_RAW_MEMBER_BYTES = 1024 * 1024
MAX_RAW_UNCOMPRESSED_BYTES = 4 * 1024 * 1024
MAX_PLAN_BYTES = 512 * 1024
MAX_BANK_BYTES = 8 * 1024 * 1024
MAX_REPLAY_BYTES = 8 * 1024 * 1024
MAX_STRESS_SHARD_BYTES = 16 * 1024 * 1024
MAX_STRESS_REPORT_BYTES = 2 * 1024 * 1024
MAX_ANCHOR_DISPLACEMENT = 12
REPAIR_MIN_DRIVER_PIXELS = 12
REPAIR_ANCHOR_SUPPORT_PIXELS = 4
REPAIR_ANCHOR_SUPPORT_RADIUS = 2
STRESS_SHARD_COUNT = 16
STRESS_MAX_ATTEMPTS = 3
STRESS_WORKERS = 2
STRESS_TIMEOUT_SECONDS = 900
DISK_FLOOR_BYTES = 100 * 1024**3
