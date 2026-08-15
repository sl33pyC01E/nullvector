from __future__ import annotations

from dataclasses import replace

from .contract import AppendageGene, ComponentGene, DevelopmentalGenome, FAMILIES, TRAITS


def _traits(**values: float) -> tuple[float, ...]:
    defaults = {
        "size": .52, "symmetry": .90, "segmentation": .45, "stiffness": .55,
        "elasticity": .55, "bone_density": .62, "muscle_density": .64,
        "muscle_strength": .62, "neural_density": .52, "vascularity": .58,
        "metabolism": .55, "regeneration": .42, "grip": .58,
        "sensory_range": .58, "phase_coherence": .10,
    }
    defaults.update(values)
    return tuple(defaults[name] for name in TRAITS)


def _delta(**values: float) -> tuple[float, ...]:
    return tuple(values.get(name, 0.0) for name in TRAITS)


def _mix(family: int, secondary: int | None = None, amount: float = 0.0) -> tuple[float, ...]:
    result = [0.0] * len(FAMILIES)
    result[family] = 1.0 - amount
    if secondary is not None:
        result[secondary] = amount
    return tuple(result)


def _pair(
    kind: str,
    root: str,
    y: float,
    x: float,
    endpoint_y: float,
    prefix: str,
    phase: float = 0.0,
    segments: int = 2,
    endpoint_x: float | None = None,
) -> tuple[AppendageGene, AppendageGene]:
    left = f"{prefix}_l"
    right = f"{prefix}_r"
    tip_x = x * 2.1 if endpoint_x is None else endpoint_x
    return (
        AppendageGene(left, kind, root, (-x, y), (-tip_x, endpoint_y), segments, -1, phase, -1, right, _delta(muscle_strength=.08, grip=.06)),
        AppendageGene(right, kind, root, (x, y), (tip_x, endpoint_y), segments, 1, (phase + .5) % 1.0, 1, left, _delta(muscle_strength=.08, grip=.06)),
    )


def _base_genomes() -> tuple[DevelopmentalGenome, ...]:
    humanoid_components = (
        ComponentGene("torso", "soma", (0,-2), (4.6,5.5), organ="heart", trait_delta=_delta(muscle_density=.08)),
        ComponentGene("pelvis", "pelvis", (0,4), (3.8,3.0), "torso", organ="gut", trait_delta=_delta(bone_density=.10)),
        ComponentGene("head", "head", (0,-10), (3.2,3.2), "torso", organ="brain", trait_delta=_delta(neural_density=.22,sensory_range=.16)),
        ComponentGene("lungs", "respirator", (0,-3), (2.4,2.0), "torso", organ="lung", trait_delta=_delta(vascularity=.15,metabolism=.08)),
        ComponentGene("eyes", "sensor_crown", (0,-12), (2.0,1.0), "head", organ="eye", trait_delta=_delta(sensory_range=.28)),
    )
    humanoid_appendages = _pair("arm","torso",-2,3.5,5,"arm",0) + _pair("leg","pelvis",1,2.2,15,"leg",.5)
    animal_components = (
        ComponentGene("chest", "soma", (0,0), (7.4,3.8), organ="lung", trait_delta=_delta(muscle_density=.12)),
        ComponentGene("haunch", "pelvis", (0,3), (6.8,3.9), "chest", organ="gut", trait_delta=_delta(muscle_strength=.14)),
        ComponentGene("neck", "soma", (0,-3), (3.0,2.8), "chest", organ="none", trait_delta=_delta(elasticity=.10)),
        ComponentGene("head", "head", (0,-7), (4.4,3.0), "neck", organ="brain", trait_delta=_delta(neural_density=.15)),
        ComponentGene("muzzle", "mouth", (0,-9), (3.0,1.6), "head", organ="jaw", trait_delta=_delta(grip=.20)),
        ComponentGene("eyes", "sensor_crown", (0,-8), (3.3,1.0), "head", organ="eye", trait_delta=_delta(sensory_range=.25)),
    )
    animal_appendages = _pair("leg","chest",1,5.7,11,"foreleg",0,3,8.6) + _pair("leg","haunch",1.5,3.6,13,"hindleg",.5,3,4.5) + (
        # Dorsal tail: intentionally above the locomotor plane so it cannot read
        # as a fifth leg in the vertically locked 2.5D presentation.
        AppendageGene("tail", "tail", "chest", (4.8,-2.2), (14,-5), 5, 1, .25, 1, None, _delta(elasticity=.24)),
    )
    plant_components = (
        ComponentGene("bulb", "soma", (0,3), (6.8,5.0), organ="bulb", trait_delta=_delta(storage=.0,regeneration=.25)),
        ComponentGene("stem", "soma", (0,-5), (2.5,8.0), "bulb", organ="vascular", trait_delta=_delta(stiffness=.12,vascularity=.24)),
        ComponentGene("crown", "sensor_crown", (0,-13), (8.8,3.5), "stem", organ="photoreceptor", trait_delta=_delta(sensory_range=.16,metabolism=-.12)),
        ComponentGene("meristem", "neural_cluster", (0,-8), (2.0,2.0), "stem", organ="meristem", trait_delta=_delta(neural_density=.15,regeneration=.25)),
    )
    plant_appendages = _pair("root","bulb",2,4.8,16,"root_a",0,3,10.5) + _pair("root","bulb",3,2.0,18,"root_b",.5,3,4.0) + _pair("frond","crown",0,6.2,-9,"frond",.25,3,13.5)
    anomaly_components = (
        ComponentGene("core", "soma", (0,-2), (7.2,7.2), organ="phase_brain", trait_delta=_delta(phase_coherence=.45,elasticity=.12)),
        ComponentGene("sensor", "sensor_crown", (0,-11), (2.8,2.5), "core", organ="singularity", trait_delta=_delta(sensory_range=.28,phase_coherence=.20)),
        ComponentGene("orbital_l", "orbital", (-7,-2), (2.7,2.7), "core", -1, _delta(phase_coherence=.18), "orbital"),
        ComponentGene("orbital_r", "orbital", (7,-2), (2.7,2.7), "core", 1, _delta(phase_coherence=.18), "orbital"),
        ComponentGene("transmuter", "gut", (0,3), (2.2,2.0), "core", organ="transmuter", trait_delta=_delta(metabolism=-.18,phase_coherence=.12)),
    )
    anomaly_appendages = (
        _pair("tendril","core",1,5.4,10,"outer_tendril",0,4,12.0)
        + _pair("tendril","core",3,3.2,15,"middle_tendril",.33,4,7.0)
        + _pair("tendril","core",4,1.4,18,"inner_tendril",.66,4,2.8)
    )
    machine_components = (
        ComponentGene("hull", "soma", (0,-2), (8.4,4.8), organ="processor", trait_delta=_delta(stiffness=.22,bone_density=.20)),
        ComponentGene("drive", "pelvis", (0,4), (7.2,3.2), "hull", organ="battery", trait_delta=_delta(muscle_strength=.12)),
        ComponentGene("mast", "head", (0,-10), (2.0,5.5), "hull", organ="optic", trait_delta=_delta(sensory_range=.25)),
        ComponentGene("coolant", "circulator", (0,0), (2.2,2.2), "hull", organ="coolant_pump", trait_delta=_delta(vascularity=.15)),
        ComponentGene("armor", "armor", (0,-3), (9.0,3.0), "hull", organ="none", trait_delta=_delta(stiffness=.18)),
    )
    machine_appendages = _pair("wheel","drive",1,5.2,14,"drive_wheel",0,2,7.8) + _pair("hardpoint","hull",-1,7.0,-1,"hardpoint",.25,2,13.0)
    return (
        DevelopmentalGenome("base_humanoid",0xD300, _mix(0), _traits(), humanoid_components, humanoid_appendages),
        DevelopmentalGenome("base_animalian",0xD301, _mix(1), _traits(size=.58,muscle_density=.72,muscle_strength=.72,elasticity=.62), animal_components, animal_appendages),
        DevelopmentalGenome("base_plantlike",0xD302, _mix(2), _traits(symmetry=.94,stiffness=.60,bone_density=.35,muscle_density=.34,metabolism=.35,regeneration=.76), plant_components, plant_appendages),
        DevelopmentalGenome("base_anomaly",0xD303, _mix(3), _traits(symmetry=.70,stiffness=.32,bone_density=.18,muscle_density=.42,elasticity=.82,phase_coherence=.82), anomaly_components, anomaly_appendages),
        DevelopmentalGenome("base_machine",0xD304, _mix(4), _traits(stiffness=.82,bone_density=.86,muscle_density=.52,muscle_strength=.75,metabolism=.25,regeneration=.24), machine_components, machine_appendages),
    )


def review_genomes() -> tuple[DevelopmentalGenome, ...]:
    base = _base_genomes()
    humanoid, animal, plant, anomaly, machine = base
    # The second row demonstrates that family is a prior rather than a cage:
    # grafted modules bring their local trait sources and diffuse into neighbors.
    grafted = (
        replace(humanoid, genome_id="graft_humanoid_rooted", seed=0xD310, family_mix=_mix(0,2,.18), parent_ids=(humanoid.genome_id,plant.genome_id), appendages=humanoid.appendages + tuple(replace(a, appendage_id="g_"+a.appendage_id, root_component="pelvis", paired_with=("g_"+a.paired_with if a.paired_with else None)) for a in plant.appendages[:2])),
        replace(animal, genome_id="graft_animal_armored", seed=0xD311, family_mix=_mix(1,4,.22), parent_ids=(animal.genome_id,machine.genome_id), components=animal.components + (replace(machine.components[4], component_id="g_armor", parent="chest", anchor=(0,-3), radius=(5.8,2.8)),), appendages=animal.appendages + tuple(replace(a, appendage_id="g_"+a.appendage_id, root_component="chest", paired_with=("g_"+a.paired_with if a.paired_with else None)) for a in machine.appendages[2:])),
        replace(plant, genome_id="graft_plant_orbital", seed=0xD312, family_mix=_mix(2,3,.20), parent_ids=(plant.genome_id,anomaly.genome_id), components=plant.components + (replace(anomaly.components[2], component_id="g_orbital_l", parent="stem"), replace(anomaly.components[3], component_id="g_orbital_r", parent="stem"))),
        replace(anomaly, genome_id="graft_anomaly_muscular", seed=0xD313, family_mix=_mix(3,1,.22), parent_ids=(anomaly.genome_id,animal.genome_id), appendages=anomaly.appendages + tuple(replace(a, appendage_id="g_"+a.appendage_id, root_component="core", paired_with=("g_"+a.paired_with if a.paired_with else None)) for a in animal.appendages[:2])),
        replace(machine, genome_id="graft_machine_neural", seed=0xD314, family_mix=_mix(4,0,.20), parent_ids=(machine.genome_id,humanoid.genome_id), components=machine.components + (replace(humanoid.components[2], component_id="g_neural_head", parent="mast", anchor=(0,-13), radius=(2.7,2.7)), replace(humanoid.components[4], component_id="g_eyes", parent="g_neural_head", anchor=(0,-15), radius=(1.8,1.0))),),
    )
    return tuple(item for pair in zip(base, grafted, strict=True) for item in pair)
