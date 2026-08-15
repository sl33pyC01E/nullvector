from __future__ import annotations

from dataclasses import replace

from ..creature_stage_developmental.contract import (
    AppendageGene,
    ComponentGene,
    DevelopmentalGenome,
    FAMILIES,
    TRAITS,
)
from ..creature_stage_developmental.genomes import review_genomes


def _traits(**values: float) -> tuple[float, ...]:
    base = {
        "size": .52, "symmetry": .96, "segmentation": .50, "stiffness": .55,
        "elasticity": .55, "bone_density": .62, "muscle_density": .64,
        "muscle_strength": .62, "neural_density": .52, "vascularity": .58,
        "metabolism": .55, "regeneration": .42, "grip": .58,
        "sensory_range": .58, "phase_coherence": .10,
    }
    base.update(values)
    return tuple(base[name] for name in TRAITS)


def _delta(**values: float) -> tuple[float, ...]:
    return tuple(values.get(name, 0.0) for name in TRAITS)


def _mix(index: int) -> tuple[float, ...]:
    values = [0.0] * len(FAMILIES)
    values[index] = 1.0
    return tuple(values)


def _pair(
    kind: str,
    root: str,
    root_x: float,
    root_y: float,
    tip_x: float,
    tip_y: float,
    prefix: str,
    *,
    segments: int = 3,
    phase: float = 0.0,
    delta: tuple[float, ...] | None = None,
) -> tuple[AppendageGene, AppendageGene]:
    left, right = f"{prefix}_l", f"{prefix}_r"
    local = delta or _delta(muscle_strength=.08, grip=.06)
    return (
        AppendageGene(left, kind, root, (-root_x, root_y), (-tip_x, tip_y), segments, -1, phase, -1, right, local),
        AppendageGene(right, kind, root, (root_x, root_y), (tip_x, tip_y), segments, 1, (phase + .5) % 1.0, 1, left, local),
    )


def _humanoids() -> tuple[DevelopmentalGenome, ...]:
    # Preserve the accepted humanoid prior; variants change proportions in
    # bilateral pairs rather than changing its anatomical vocabulary.
    base = review_genomes()[0]
    variants = []
    specs = (
        ("balanced", 0xE200, 1.00, 1.00),
        ("stout", 0xE201, 1.16, .90),
        ("runner", 0xE202, .91, 1.15),
        ("heavy", 0xE203, 1.24, 1.05),
        ("fine", 0xE204, .84, 1.04),
        ("tall", 0xE205, .94, 1.22),
    )
    for label, seed, width, height in specs:
        components = tuple(replace(c, anchor=(c.anchor[0] * width, c.anchor[1] * height), radius=(c.radius[0] * width, c.radius[1] * min(height, 1.12))) for c in base.components)
        appendages = tuple(replace(a, root_offset=(a.root_offset[0] * width, a.root_offset[1] * height), endpoint=(a.endpoint[0] * width, a.endpoint[1] * height)) for a in base.appendages)
        variants.append(replace(base, genome_id=f"v2_humanoid_{label}", seed=seed, traits=_traits(symmetry=.98), components=components, appendages=appendages))
    return tuple(variants)


def _animal(label: str, seed: int, *, body_w: float, body_h: float, leg_spread: float, leg_len: float, head_w: float, grasper: bool, plated: bool = False) -> DevelopmentalGenome:
    components = [
        ComponentGene("thorax", "soma", (0,-1.0), (body_w,body_h), organ="lung", trait_delta=_delta(muscle_density=.14)),
        ComponentGene("belly", "pelvis", (0,2.0), (body_w*.82,body_h*.78), "thorax", organ="gut", trait_delta=_delta(metabolism=.12)),
        ComponentGene("heart", "circulator", (0,-.4), (2.0,1.8), "thorax", organ="heart", trait_delta=_delta(vascularity=.20)),
        ComponentGene("head", "head", (0,-body_h-2.0), (head_w,2.5), "thorax", organ="brain", trait_delta=_delta(neural_density=.18)),
        ComponentGene("muzzle", "mouth", (0,-body_h-4.0), (head_w*.62,1.2), "head", organ="jaw", trait_delta=_delta(grip=.22)),
        ComponentGene("eyes", "sensor_crown", (0,-body_h-3.1), (head_w*.76,.8), "head", organ="eye", trait_delta=_delta(sensory_range=.25)),
    ]
    if plated:
        components.append(ComponentGene("carapace", "armor", (0,-2.0), (body_w*1.04,body_h*.62), "thorax", organ="none", trait_delta=_delta(stiffness=.20)))
    appendages = (
        _pair("leg","thorax",body_w*.68,.2,leg_spread,leg_len,"foreleg",segments=3,phase=0.0)
        + _pair("leg","belly",body_w*.42,1.3,leg_spread*.62,leg_len+1.4,"hindleg",segments=3,phase=.5)
    )
    if grasper:
        appendages += (AppendageGene("dorsal_grasper", "tail", "thorax", (0,-body_h*.55), (0,-body_h-9.0), 3, 0, .25, 1, None, _delta(elasticity=.16,grip=.20,sensory_range=.08)),)
    return DevelopmentalGenome(
        f"v2_animalian_{label}", seed, _mix(1),
        _traits(size=.58,symmetry=.99,muscle_density=.74,muscle_strength=.72,elasticity=.62),
        tuple(components), appendages,
    )


def _animals() -> tuple[DevelopmentalGenome, ...]:
    return (
        _animal("runner",0xE210,body_w=7.8,body_h=3.0,leg_spread=8.8,leg_len=11.5,head_w=3.5,grasper=False),
        _animal("grazer",0xE211,body_w=9.0,body_h=3.5,leg_spread=9.5,leg_len=10.5,head_w=4.0,grasper=False),
        _animal("climber",0xE212,body_w=7.0,body_h=3.1,leg_spread=10.5,leg_len=10.0,head_w=3.2,grasper=True),
        _animal("burrower",0xE213,body_w=9.5,body_h=2.8,leg_spread=10.8,leg_len=8.5,head_w=4.7,grasper=False),
        _animal("plated",0xE214,body_w=8.7,body_h=3.7,leg_spread=9.4,leg_len=10.0,head_w=3.6,grasper=False,plated=True),
        _animal("grasper",0xE215,body_w=7.9,body_h=3.2,leg_spread=8.8,leg_len=11.0,head_w=3.4,grasper=True),
    )


def _plants() -> tuple[DevelopmentalGenome, ...]:
    base = review_genomes()[4]
    specs = (("bulb",1.08,.92),("reed",.78,1.22),("crown",1.18,1.00),("rooted",1.00,1.12),("compact",.88,.84),("tower",.90,1.28))
    result = []
    for ordinal, (label,width,height) in enumerate(specs):
        components = tuple(replace(c, anchor=(c.anchor[0]*width,c.anchor[1]*height), radius=(c.radius[0]*width,c.radius[1]*height)) for c in base.components)
        appendages = tuple(replace(a, root_offset=(a.root_offset[0]*width,a.root_offset[1]*height), endpoint=(a.endpoint[0]*width,a.endpoint[1]*height)) for a in base.appendages)
        result.append(replace(base, genome_id=f"v2_plantlike_{label}", seed=0xE220+ordinal, traits=replace(base).traits, components=components, appendages=appendages))
    return tuple(result)


def _anomaly(label: str, seed: int, radius: float, fibers: int, orbital_scale: float) -> DevelopmentalGenome:
    components = [
        ComponentGene("core", "soma", (0,-1.5), (radius,radius), organ="phase_brain", trait_delta=_delta(phase_coherence=.50,elasticity=.16)),
        ComponentGene("singularity", "sensor_crown", (0,-1.5), (2.4,2.4), "core", organ="singularity", trait_delta=_delta(sensory_range=.30)),
        ComponentGene("orbital_l", "orbital", (-radius*.92,-1.5), (orbital_scale,orbital_scale), "core", -1, _delta(phase_coherence=.20), "orbital"),
        ComponentGene("orbital_r", "orbital", (radius*.92,-1.5), (orbital_scale,orbital_scale), "core", 1, _delta(phase_coherence=.20), "orbital"),
        ComponentGene("transmuter", "gut", (0,2.2), (2.2,2.0), "core", organ="transmuter", trait_delta=_delta(metabolism=-.20)),
    ]
    appendages: tuple[AppendageGene, ...] = ()
    spreads = ((radius*.92,12.5),(radius*.68,14.5),(radius*.40,16.0),(radius*1.08,9.5))
    for index in range(fibers):
        root_x, tip_y = spreads[index]
        appendages += _pair("tendril","core",root_x,1.5+index*.55,root_x*(1.25-index*.12),tip_y,f"fiber_{index}",segments=4,phase=(index*.19)%1.0,delta=_delta(elasticity=.24,phase_coherence=.16))
    return DevelopmentalGenome(f"v2_anomaly_{label}",seed,_mix(3),_traits(symmetry=.96,stiffness=.25,bone_density=.12,muscle_density=.38,elasticity=.88,phase_coherence=.88),tuple(components),appendages)


def _anomalies() -> tuple[DevelopmentalGenome, ...]:
    return tuple(_anomaly(*spec) for spec in (
        ("orb",0xE230,7.5,3,2.5),("fibrous",0xE231,7.0,4,2.2),("dense",0xE232,8.2,4,2.8),
        ("small",0xE233,6.2,3,2.0),("halo",0xE234,7.4,2,3.2),("manylimb",0xE235,6.8,4,2.4),
    ))


def _machine(label: str, seed: int, *, width: float, tracks: bool, walker: bool, hardpoints: int) -> DevelopmentalGenome:
    components = (
        ComponentGene("hull","soma",(0,-2),(width,4.2),organ="processor",trait_delta=_delta(stiffness=.24,bone_density=.22)),
        ComponentGene("drive","pelvis",(0,3),(width*.88,2.8),"hull",organ="battery",trait_delta=_delta(muscle_strength=.12)),
        ComponentGene("mast","head",(0,-8),(2.2,3.8),"hull",organ="optic",trait_delta=_delta(sensory_range=.28)),
        ComponentGene("sensor_bar","sensor_crown",(0,-11),(4.2,1.0),"mast",organ="optic",trait_delta=_delta(sensory_range=.24)),
        ComponentGene("coolant","circulator",(0,-1),(2.4,2.2),"hull",organ="coolant_pump",trait_delta=_delta(vascularity=.16)),
        ComponentGene("armor","armor",(0,-3.5),(width+1.0,2.0),"hull",organ="none",trait_delta=_delta(stiffness=.20)),
    )
    appendages: tuple[AppendageGene,...] = ()
    if tracks:
        appendages += _pair("wheel","drive",width*.60,1.0,width*.82,10.0,"track",segments=2,phase=0.0)
    if walker:
        appendages += _pair("leg","drive",width*.52,1.0,width*.72,13.0,"walker",segments=3,phase=.5)
    for index in range(hardpoints):
        spread = width * (.68 + index*.20)
        appendages += _pair("hardpoint","hull",spread,-1.0-index,spread*1.45,-2.0-index,f"hardpoint_{index}",segments=2,phase=.25+index*.1)
    return DevelopmentalGenome(f"v2_machine_{label}",seed,_mix(4),_traits(symmetry=.99,stiffness=.88,bone_density=.88,muscle_density=.48,muscle_strength=.76,metabolism=.24,regeneration=.20),components,appendages)


def _machines() -> tuple[DevelopmentalGenome, ...]:
    return (
        _machine("tracked",0xE240,width=8.5,tracks=True,walker=False,hardpoints=1),
        _machine("walker",0xE241,width=7.2,tracks=False,walker=True,hardpoints=1),
        _machine("hybrid",0xE242,width=9.0,tracks=True,walker=True,hardpoints=1),
        _machine("artillery",0xE243,width=9.5,tracks=True,walker=False,hardpoints=2),
        _machine("utility",0xE244,width=7.8,tracks=False,walker=True,hardpoints=2),
        _machine("compact",0xE245,width=6.5,tracks=True,walker=False,hardpoints=1),
    )


def morphology_review_genomes() -> tuple[DevelopmentalGenome, ...]:
    """Thirty family-balanced, symmetry-first priors for human review."""
    return _humanoids() + _animals() + _plants() + _anomalies() + _machines()
