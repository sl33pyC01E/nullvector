from __future__ import annotations

import json

from .evaluation import evaluate


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
