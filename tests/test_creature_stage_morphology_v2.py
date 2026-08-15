from __future__ import annotations

import json

from forge.creature_stage_developmental.contract import FAMILIES
from forge.creature_stage_developmental.development import develop
from forge.creature_stage_morphology_v2.genomes import morphology_review_genomes
from forge.creature_stage_morphology_v2.review import REQUIRED_ORGANS, _symmetry, build_review
from forge.creature_stage_grounded_locomotion.physics import locomotor_modes, primary_mode


def test_balanced_family_bank_and_anatomy() -> None:
    genomes=morphology_review_genomes()
    assert len(genomes)==30
    for family_index,family in enumerate(FAMILIES):
        rows=genomes[family_index*6:(family_index+1)*6]
        assert all(row.family_mix[family_index]==1.0 for row in rows)
        for genome in rows:
            organism=develop(genome)
            assert _symmetry(organism)>=.90
            organs={component.organ for component in genome.components}
            assert REQUIRED_ORGANS[family] <= organs


def test_animal_machine_anomaly_shape_grammar() -> None:
    genomes=morphology_review_genomes()
    animals=genomes[6:12]; anomalies=genomes[18:24]; machines=genomes[24:30]
    for genome in animals:
        leg_pairs=[a for a in genome.appendages if a.kind=="leg"]
        assert len(leg_pairs)==4
        assert all(a.endpoint[1]>7 for a in leg_pairs)
        tails=[a for a in genome.appendages if a.kind=="tail"]
        assert all(a.side==0 and a.endpoint[0]==0 and a.endpoint[1]<0 for a in tails)
    for genome in anomalies:
        core=next(c for c in genome.components if c.component_id=="core")
        assert core.radius[0]==core.radius[1]
        assert len([a for a in genome.appendages if a.kind=="tendril"])>=4
    for genome in machines:
        assert any(a.kind in {"wheel","leg"} for a in genome.appendages)
        assert any(a.kind=="hardpoint" for a in genome.appendages)


def test_review_is_atomic_and_manifested(tmp_path) -> None:
    manifest_path=build_review(tmp_path/"review")
    payload=json.loads(manifest_path.read_text(encoding="ascii"))
    assert payload["count"]==30
    assert all(payload["gates"].values())
    assert len(payload["artifacts"])==6


def test_locomotion_is_component_owned_not_family_hardcoded() -> None:
    genomes=morphology_review_genomes()
    expected=("step","step","drag","float","wheel")
    for genome,wanted in zip((genomes[0],genomes[6],genomes[12],genomes[18],genomes[24]),expected,strict=True):
        organism=develop(genome)
        assert primary_mode(organism,locomotor_modes(organism))==wanted
