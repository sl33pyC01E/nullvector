"""CPU-first neural topology representation and deterministic repair foundation.

The package initializer intentionally avoids importing PyTorch. Corpus auditing and
compilation remain lightweight and cannot accidentally initialize an accelerator.
Import ``forge.map_topology_neural.codec`` explicitly for the CPU VQ smoke model.
"""

__all__ = [
    "artifacts",
    "checkpoint",
    "codec",
    "compiler",
    "contract",
    "corpus",
    "render",
    "smoke",
]
