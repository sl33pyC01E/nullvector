from __future__ import annotations

from forge.nature_sim_v2 import AdventureState, NatureWorld, founder_genomes
from forge.qud_items_v1 import RECIPES, craft, generate_artifact


def test_relic_generation_is_combinatorial_and_deterministic() -> None:
    left = generate_artifact(seed=91, provenance="vault-a")
    right = generate_artifact(seed=91, provenance="vault-a")
    assert left == right
    assert left.semantic_sha256() == right.semantic_sha256()
    bank = [generate_artifact(seed=seed, provenance=f"vault-{seed}") for seed in range(128)]
    assert len({item.semantic_sha256() for item in bank}) == 128
    assert len({item.name for item in bank}) > 80
    assert {item.slot for item in bank} == {"carapace", "manipulator", "sensor", "core"}


def test_recipe_consumes_material_and_produces_biased_equipment() -> None:
    inventory = {"metal": 10.0, "crystal": 10.0, "biomass": 10.0, "rock": 10.0, "water": 10.0}
    recipe = RECIPES[0]
    result = craft(recipe, seed=5, provenance="test-craft", inventory=inventory)
    assert result.slot == recipe.slot
    assert set(recipe.required_components).issubset(result.components)
    assert result.effect("damage") > 0 and result.effect("harvest") > 0
    assert inventory["metal"] == 8.6 and inventory["crystal"] == 9.65


def test_adventure_equipment_changes_live_play_capabilities() -> None:
    adventure = AdventureState(seed=17, size=64)
    adventure.inventory.update({"metal": 5, "crystal": 5, "biomass": 5, "rock": 5, "water": 5})
    adventure.recipe_index = 0
    message = adventure.craft_selected()
    assert "CRAFTED" in message and len(adventure.artifacts) == 1
    assert adventure.bonus("damage") > 0 and adventure.bonus("harvest") > 0
    before = adventure.equipped_artifacts()[0].durability
    adventure.abrade("manipulator", .1)
    assert adventure.equipped_artifacts()[0].durability < before


def test_relic_site_yields_seeded_equipment_once() -> None:
    world = NatureWorld(seed=22, size=64)
    entity_id = world.add_organism(founder_genomes(variants_per_family=1)[0], (8, 8), energy=.8)
    adventure = AdventureState(seed=19, size=64)
    vault = next(site for site in adventure.sites if site.kind == "relic_vault")
    world.organisms[entity_id].position[:] = vault.position
    assert "RELIC" in adventure.interact(world, world.organisms[entity_id])
    assert len(adventure.artifacts) == 1 and adventure.equipped
    adventure.interact(world, world.organisms[entity_id])
    assert len(adventure.artifacts) == 1
