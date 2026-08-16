from pathlib import Path
from .review import build_review

if __name__=="__main__":
    report=build_review(Path("outputs/creature_stage_grounded_locomotion_25d/review_v1"))
    print(f"GROUNDED_25D {'PASS' if report['passed'] else 'FAIL'} {report['metrics']}")

