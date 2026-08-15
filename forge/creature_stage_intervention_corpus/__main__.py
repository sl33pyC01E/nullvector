from __future__ import annotations

import json
import sys

from .validation import InterventionCorpusValidationError, validate_intervention_corpus


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m forge.creature_stage_intervention_corpus CORPUS", file=sys.stderr)
        return 2
    try:
        report = validate_intervention_corpus(sys.argv[1])
    except InterventionCorpusValidationError as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
