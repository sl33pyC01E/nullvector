from __future__ import annotations
from forge.world_frame_refiner_v2.contract import Plan
def test_refiner_schedule_is_segmented():assert Plan().updates%Plan().segment==0
