from __future__ import annotations

import numpy as np

from forge.nature_sim_v2 import founder_genomes, phenotype_traits, phenotype_vector, recombine


def test_founders_expose_readable_heritable_phenotypes() -> None:
    genomes = founder_genomes(variants_per_family=3)
    records = [phenotype_traits(genome) for genome in genomes]
    assert all(len(items) >= 5 for items in records)
    assert len({tuple(item.key for item in items) for items in records}) >= 5
    assert any(item.key == "wheeled" for items in records for item in items)
    assert any(item.key == "root_drag" for items in records for item in items)


def test_phenotype_vector_and_traits_change_through_breeding() -> None:
    founders = founder_genomes(variants_per_family=2)
    child = recombine(founders[0], founders[1], seed=1234)
    vector = phenotype_vector(child)
    assert vector.shape == (44,) and vector.dtype == np.float32 and np.isfinite(vector).all()
    assert child.developmental.generation == 1
    assert not np.array_equal(vector, phenotype_vector(founders[0]))
    assert phenotype_traits(child)
