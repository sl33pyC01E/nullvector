extends "res://scripts/cellular_motion_lab.gd"

const ECOLOGY_FORMAT := "nullvector-cellular-ecology-native-catalog-v3"
const ECOLOGY_ORIGIN := Vector2(505.0, 99.0)
const ECOLOGY_CELL := 12.0
const ECOLOGY_COLORS := [Color("#3fe0ff"), Color("#ff8746"), Color("#6eff56"), Color("#db53ff"), Color("#aabfe1")]

@export_file("*.json") var ecology_catalog_path := "res://generated/cellular_ecology/v3/ecology_catalog.json"

var ecology_catalog: Dictionary = {}
var ecology_map_index := 0
var ecology_label: Label
var habitat_label: Label
var ecology_damage_accumulator := 0.0


func _ready() -> void:
	ecology_catalog = _load_json(ecology_catalog_path)
	_validate_ecology_catalog()
	super()
	_seed_ecology_population()
	_build_ecology_overlay()
	_reset_ecology_resources()
	_refresh_ecology_overlay()
	if not startup_errors.is_empty():
		status_label.text = "FAIL-CLOSED // " + ", ".join(startup_errors)
		status_label.modulate = Color("#ff526d")
	if "--cellular-ecology-smoke" in OS.get_cmdline_user_args(): call_deferred("_run_ecology_smoke")


func _uses_autonomous_motion() -> bool:
	return true


func _seed_ecology_population() -> void:
	var present: Dictionary = {}
	for organism in organisms: present[str(int(organism["data"].get("family_id", -1)))] = true
	var positions := [Vector2(620, 230), Vector2(760, 390), Vector2(885, 225), Vector2(1010, 390), Vector2(1130, 235)]
	for family_id in range(5):
		if present.has(str(family_id)): continue
		var species_index := -1
		for index in range(catalog.get("species", []).size()):
			if int(catalog.get("species", [])[index].get("family_id", -1)) == family_id: species_index = index; break
		if species_index < 0: startup_errors.append("ecology family species %d" % family_id); continue
		var data := _load_species_data(species_index)
		if data.is_empty(): startup_errors.append("ecology family runtime %d" % family_id); continue
		var organism := _create_organism(data, positions[family_id], 0, 0); organisms.append(organism)
		_event("ECOLOGY BIRTH // FAMILY %d // %s" % [family_id, str(data.get("sample_id", "?"))], ECOLOGY_COLORS[family_id])
	if organisms.size() != 5: startup_errors.append("ecology initial family population")


func _validate_ecology_catalog() -> void:
	if ecology_catalog.get("format", "") != ECOLOGY_FORMAT: startup_errors.append("ecology format")
	if ecology_catalog.get("status", "") != "ready": startup_errors.append("ecology status")
	if int(ecology_catalog.get("map_count", -1)) != 6: startup_errors.append("ecology map census")
	if int(ecology_catalog.get("resource_node_count", -1)) != 120: startup_errors.append("ecology resource census")
	if ecology_catalog.get("maps", []).size() != 6: startup_errors.append("ecology map records")
	for map_record in ecology_catalog.get("maps", []):
		if map_record.get("fields_u8", {}).size() != 7: startup_errors.append("ecology fields")
		if map_record.get("family_suitability_u8", []).size() != 5: startup_errors.append("ecology niches")
		if map_record.get("resource_nodes", []).size() != 20: startup_errors.append("ecology nodes")


func _build_ecology_overlay() -> void:
	var canvas := CanvasLayer.new(); add_child(canvas)
	var panel := _panel(canvas, Rect2(850, 174, 384, 78))
	ecology_label = _label(panel, Vector2(12, 8), Vector2(360, 22), "ECOLOGY", LIME, 10)
	habitat_label = _label(panel, Vector2(12, 31), Vector2(360, 39), "HABITAT", MUTED, 8)
	controls_label.text += "\nA/D habitat // ecological food regrows"


func _current_ecology() -> Dictionary:
	var maps: Array = ecology_catalog.get("maps", [])
	return maps[ecology_map_index] if ecology_map_index >= 0 and ecology_map_index < maps.size() else {}


func _grid_to_world(point: Array) -> Vector2:
	return ECOLOGY_ORIGIN + Vector2(float(point[0]) + 0.5, float(point[1]) + 0.5) * ECOLOGY_CELL


func _world_to_index(position: Vector2) -> int:
	var local := (position - ECOLOGY_ORIGIN) / ECOLOGY_CELL
	var x: int = clampi(int(floor(local.x)), 0, 47); var y: int = clampi(int(floor(local.y)), 0, 47)
	return y * 48 + x


func _field(name: String, position: Vector2) -> float:
	var values: Array = _current_ecology().get("fields_u8", {}).get(name, [])
	var index := _world_to_index(position)
	return float(values[index]) / 255.0 if index >= 0 and index < values.size() else 0.0


func _reset_ecology_resources() -> void:
	foods = foods.filter(func(food: Dictionary): return not food.has("ecology_id"))
	for node in _current_ecology().get("resource_nodes", []):
		foods.append({"position": _grid_to_world(node.get("position", [0, 0])), "amount": float(node.get("capacity", 20.0)), "capacity": float(node.get("capacity", 20.0)), "regrowth": float(node.get("regrowth_per_second", 0.1)), "pulse": 0.0, "ecology_id": int(node.get("id", -1)), "family_id": int(node.get("family_id", 0))})
	queue_redraw()


func _resource_affinity(organism: Dictionary, node: Dictionary) -> float:
	var family := int(organism["data"].get("family_id", 0)); var resource_family := int(node.get("family_id", -1))
	if resource_family < 0: return 1.0 if family in [0, 1] else 0.05
	var matrix := [
		[1.00, 0.84, 0.78, 0.42, 0.68],
		[0.88, 1.00, 0.66, 0.06, 0.08],
		[0.12, 0.18, 1.00, 0.24, 0.04],
		[0.16, 0.04, 0.22, 1.00, 0.58],
		[0.50, 0.03, 0.08, 0.72, 1.00],
	]
	return float(matrix[clampi(family, 0, 4)][clampi(resource_family, 0, 4)])


func _is_resource_cell(organism: Dictionary, index: int) -> bool:
	var family := int(organism["data"].get("family_id", 0)); var flags := int(organism["flags"][index]); var tissue := int(organism["tissue"][index])
	if family in [0, 1]: return (flags & FLAG_MOUTH) != 0
	if family == 2: return (flags & FLAG_PHOTOSYNTHETIC) != 0 or tissue in [13, 14]
	if family == 3: return (flags & FLAG_EMITTER) != 0 or tissue in [4, 12]
	return tissue in [6, 9, 12] or (flags & FLAG_EMITTER) != 0


func _step_food(delta: float) -> void:
	for food in foods:
		food["pulse"] = fmod(float(food.get("pulse", 0.0)) + delta * 3.0, TAU)
		if food.has("ecology_id"): food["amount"] = minf(float(food.get("capacity", 20.0)), float(food["amount"]) + float(food.get("regrowth", 0.1)) * delta)
	for organism in organisms:
		var family := int(organism["data"].get("family_id", 0)); organism["ecology_resource_intake"] = float(organism.get("ecology_resource_intake", 0.0))
		for cell_index in range(organism["position"].size()):
			if not organism["alive"][cell_index] or not _is_resource_cell(organism, cell_index): continue
			for food in foods:
				if float(food.get("amount", 0.0)) <= 0.0 or organism["position"][cell_index].distance_to(food["position"]) >= 13.0: continue
				var affinity := _resource_affinity(organism, food)
				if affinity < 0.10: continue
				var bite := minf(float(food["amount"]), (2.5 + 5.5 * affinity) * delta); food["amount"] = float(food["amount"]) - bite; organism["food_consumed"] = float(organism["food_consumed"]) + bite; organism["ecology_resource_intake"] = float(organism["ecology_resource_intake"]) + bite
				if family in [0, 1]: organism["nutrient"][cell_index] = float(organism["nutrient"][cell_index]) + bite * affinity
				elif family == 2:
					organism["nutrient"][cell_index] = float(organism["nutrient"][cell_index]) + bite * 0.35; organism["energy"][cell_index] = float(organism["energy"][cell_index]) + bite * 2.8 * affinity
				elif family == 3: organism["energy"][cell_index] = float(organism["energy"][cell_index]) + bite * 6.2 * affinity
				else: organism["energy"][cell_index] = float(organism["energy"][cell_index]) + bite * 5.0 * affinity
	foods = foods.filter(func(food: Dictionary): return food.has("ecology_id") or float(food.get("amount", 0.0)) > 0.01)
	var present: Dictionary = {}
	for food in foods:
		if food.has("ecology_id"): present[str(int(food["ecology_id"]))] = true
	for node in _current_ecology().get("resource_nodes", []):
		var key := str(int(node.get("id", -1)))
		if not present.has(key):
			foods.append({"position": _grid_to_world(node.get("position", [0, 0])), "amount": float(node.get("regrowth_per_second", 0.1)) * delta, "capacity": float(node.get("capacity", 20.0)), "regrowth": float(node.get("regrowth_per_second", 0.1)), "pulse": 0.0, "ecology_id": int(node.get("id", -1)), "family_id": int(node.get("family_id", 0))})


func _step_ecology_motility(organism: Dictionary, delta: float) -> void:
	var family := int(organism["data"].get("family_id", 0)); if family == 2: return
	var capacity: Dictionary = organism.get("physiology_capacities", {}); var motor := float(capacity.get("locomotion", 0.0)) * float(capacity.get("neural", 0.0))
	if motor <= 0.08: return
	var center := _organism_center(organism); var target := _best_ecology_resource(organism)
	if target.is_empty(): return
	var difference: Vector2 = target["position"] - center; if difference.length() < 9.0: return
	var speed: float = [20.0, 31.0, 0.0, 17.0, 23.0][family]; var impulse: Vector2 = difference.normalized() * speed * motor * delta
	for index in range(organism["velocity"].size()):
		if organism["alive"][index]: organism["velocity"][index] += impulse
	organism["ecology_target_id"] = int(target.get("ecology_id", -1)); organism["ecology_motive_impulse"] = float(organism.get("ecology_motive_impulse", 0.0)) + impulse.length()


func _best_ecology_resource(organism: Dictionary) -> Dictionary:
	var center := _organism_center(organism); var target: Dictionary = {}; var best := -1e20
	for food in foods:
		if float(food.get("amount", 0.0)) <= 0.01: continue
		var affinity := _resource_affinity(organism, food); var distance := center.distance_to(food["position"]); var score := affinity * minf(1.0, float(food["amount"]) / maxf(1.0, float(food.get("capacity", 20.0)))) - distance / 720.0
		if score > best: best = score; target = food
	return target


func _organism_health_fraction(organism: Dictionary) -> float:
	return clampf(_sum_float(organism.get("health", [])) / maxf(0.001, _sum_float(organism.get("max_health", []))), 0.0, 1.0)


func _organism_mean_energy(organism: Dictionary) -> float:
	var alive_count := 0; var total := 0.0
	for index in range(organism.get("alive", []).size()):
		if organism["alive"][index]: alive_count += 1; total += float(organism["energy"][index])
	return total / maxf(1.0, float(alive_count))


func _step_ecology_behavior(organism: Dictionary, delta: float) -> void:
	var capacity: Dictionary = organism.get("physiology_capacities", {})
	var family := int(organism["data"].get("family_id", 0)); var center := _organism_center(organism)
	var health := _organism_health_fraction(organism); var previous_health := float(organism.get("ecology_health_snapshot", health))
	var intake := float(organism.get("ecology_resource_intake", 0.0)); var previous_intake := float(organism.get("ecology_intake_snapshot", intake))
	var recently_hurt := previous_health - health > 0.0005; var recently_fed := intake - previous_intake > 0.0005
	var neural := float(capacity.get("neural", 0.0)); var circulation := float(capacity.get("circulation", 0.0)); var respiration := float(capacity.get("respiration", 0.0))
	var digestion := float(capacity.get("digestion", 0.0)); var sensory := float(capacity.get("sensory", 0.0)); var locomotion := float(capacity.get("locomotion", 0.0)); var immune := float(capacity.get("immune", 0.0))
	var target := _best_ecology_resource(organism); var facing := Vector2.UP; var target_distance := INF
	if not target.is_empty():
		var target_position: Vector2 = target["position"]
		facing = target_position - center; target_distance = facing.length()
	var seed_phase := float(posmod(int(organism["data"].get("seed", 0)) + family * 97, 997)) / 997.0
	var behavior_phase := fmod(simulation_time * 0.34 + seed_phase, 1.0)
	var energy := _organism_mean_energy(organism); var motion := "idle_breathe"; var behavior := "resting metabolism"; var hold := 0.0
	var locked: bool = simulation_time < float(organism.get("motion_lock_until", 0.0))
	var action_ready: bool = simulation_time >= float(organism.get("ecology_next_action_time", 0.0))
	var catastrophic: bool = circulation < 0.10 or neural < 0.10 or organism["alive"].count(true) < maxi(2, int(organism["alive"].size() * 0.24))
	if catastrophic:
		motion = "death"; behavior = "organ cascade // brain or heart offline"; hold = 0.75
	elif locked:
		motion = str(EXPECTED_MOTIONS[int(organism.get("motion_index", 0))]); behavior = str(organism.get("motion_behavior", "committed action"))
	elif recently_hurt:
		motion = "hit"; behavior = "injury reflex // connected nerves"; hold = 0.36
	elif neural < 0.38 or locomotion < 0.34:
		motion = "confused"; behavior = "motor network impaired"
	elif respiration < 0.34:
		motion = "fear"; behavior = "respiratory distress"
	elif sensory < 0.32:
		motion = "confused"; behavior = "sensory disconnect"
	elif digestion < 0.30 and family in [0, 1]:
		motion = "fear"; behavior = "digestive failure // seeking food"
	elif immune < 0.26 and health < 0.88:
		motion = "sleep"; behavior = "wound recovery impaired"
	elif family == 2:
		if recently_fed: motion = "joy"; behavior = "photosynthetic uptake"; hold = 0.50
		elif behavior_phase < 0.52: motion = "idle_breathe"; behavior = "stomata breathing"
		else: motion = "idle_wiggle"; behavior = "tropic growth sway"
	elif not target.is_empty() and target_distance > 24.0:
		motion = "locomote"; behavior = "resource pursuit"; _step_ecology_motility(organism, delta)
	elif not target.is_empty():
		if not action_ready:
			motion = "idle_wiggle" if family in [0, 3, 4] else "idle_breathe"; behavior = "action recovery // maintaining contact"
		elif family == 0:
			motion = "taunt" if behavior_phase < 0.48 else "cast"; behavior = "tool use // resource processing"; hold = 0.55
		elif family == 1:
			motion = "attack"; behavior = "feeding strike"; hold = 0.45
		elif family == 3:
			motion = "cast"; behavior = "field transduction"; hold = 0.60
		else:
			motion = "attack" if behavior_phase < 0.55 else "cast"; behavior = "ranged extraction utility"; hold = 0.52
		if action_ready: organism["ecology_next_action_time"] = simulation_time + [1.55, 1.20, 1.80, 1.75, 1.40][family] + seed_phase * 0.35
	elif energy < 0.24:
		motion = "sleep"; behavior = "energy conservation"
	elif family == 3 and behavior_phase > 0.68 and action_ready:
		motion = "cast"; behavior = "spontaneous anomaly field"; hold = 0.58
		organism["ecology_next_action_time"] = simulation_time + 2.1 + seed_phase * 0.4
	elif family == 4 and behavior_phase > 0.70 and action_ready:
		motion = "taunt"; behavior = "machine self diagnostic"; hold = 0.48
		organism["ecology_next_action_time"] = simulation_time + 1.8 + seed_phase * 0.35
	elif behavior_phase < 0.48:
		motion = "idle_breathe"; behavior = "homeostatic breathing"
	else:
		motion = "idle_wiggle"; behavior = "sensory scanning"
	_set_organism_motion(organism, motion, facing, behavior, hold)
	organism["ecology_health_snapshot"] = health; organism["ecology_intake_snapshot"] = intake
	organism["ecology_failure_state"] = {"circulation": circulation, "respiration": respiration, "digestion": digestion, "neural": neural, "sensory": sensory, "locomotion": locomotion, "immune": immune}


func _step_organism(organism: Dictionary, delta: float) -> void:
	_step_ecology_behavior(organism, delta)
	super(organism, delta)
	var center := _organism_center(organism); var toxicity := _field("toxicity", center); var temperature := _field("temperature", center); var family := int(organism["data"].get("family_id", 0))
	var toxicity_tolerance: float = [0.48, 0.42, 0.56, 0.90, 0.72][family]; var temperature_band: float = [0.32, 0.28, 0.36, 0.48, 0.44][family]
	var stress: float = maxf(0.0, toxicity - toxicity_tolerance) + maxf(0.0, absf(temperature - 0.52) - temperature_band)
	if stress > 0.0:
		ecology_damage_accumulator += stress * delta
		for index in range(organism["alive"].size()):
			if organism["alive"][index]: organism["health"][index] = maxf(0.0, float(organism["health"][index]) - stress * delta * 0.016)
	var passive_gain := 0.0
	if family == 2: passive_gain = _field("light", center) * _field("moisture", center) * (0.4 + 0.6 * _field("nutrient", center)) * 0.020
	elif family == 3: passive_gain = (_field("energy", center) * 0.75 + toxicity * 0.35) * 0.016
	elif family == 4: passive_gain = _field("energy", center) * (1.0 - _field("moisture", center) * 0.35) * 0.012
	if passive_gain > 0.0:
		var recipients := 0
		for index in range(organism["alive"].size()):
			if organism["alive"][index] and _is_resource_cell(organism, index): recipients += 1
		if recipients > 0:
			for index in range(organism["alive"].size()):
				if organism["alive"][index] and _is_resource_cell(organism, index): organism["energy"][index] = float(organism["energy"][index]) + passive_gain * delta / recipients


func _can_reproduce(organism: Dictionary) -> bool:
	if not super(organism): return false
	var center := _organism_center(organism); var biomass := _field("biomass", center); var family := int(organism["data"].get("family_id", 0)); var index := _world_to_index(center)
	var suitability_rows: Array = _current_ecology().get("family_suitability_u8", []); var suitability := float(suitability_rows[family][index]) / 255.0 if family < suitability_rows.size() and index < suitability_rows[family].size() else 0.0
	var local_support := maxf(biomass, suitability); var carrying_capacity := 1 + int(floor(local_support * 7.0))
	var threshold: float = [0.22, 0.24, 0.18, 0.20, 0.23][family]
	return suitability >= threshold and organisms.size() < carrying_capacity


func _change_ecology(delta: int) -> void:
	var maps: Array = ecology_catalog.get("maps", [])
	if maps.is_empty(): return
	ecology_map_index = posmod(ecology_map_index + delta, maps.size()); _reset_ecology_resources(); _refresh_ecology_overlay(); queue_redraw()


func _refresh_ecology_overlay() -> void:
	if ecology_label == null: return
	var record := _current_ecology(); ecology_label.text = "HABITAT // " + str(record.get("theme", "?")).to_upper()
	var stats: Dictionary = record.get("statistics", {})
	habitat_label.text = "NUTRIENT %.2f  MOISTURE %.2f  LIGHT %.2f\nTOXICITY %.2f  ENERGY %.2f  BIOMASS %.2f" % [float(stats.get("nutrient", {}).get("mean", 0.0)), float(stats.get("moisture", {}).get("mean", 0.0)), float(stats.get("light", {}).get("mean", 0.0)), float(stats.get("toxicity", {}).get("mean", 0.0)), float(stats.get("energy", {}).get("mean", 0.0)), float(stats.get("biomass", {}).get("mean", 0.0))]


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_A: _change_ecology(-1); get_viewport().set_input_as_handled(); return
		if event.keycode == KEY_D: _change_ecology(1); get_viewport().set_input_as_handled(); return
	super(event)


func _draw() -> void:
	super()
	var record := _current_ecology(); var fields: Dictionary = record.get("fields_u8", {})
	var biomass: Array = fields.get("biomass", []); var moisture: Array = fields.get("moisture", []); var toxicity: Array = fields.get("toxicity", [])
	if biomass.size() == 2304:
		for y in range(48):
			for x in range(48):
				var index := y * 48 + x; var color := Color(float(toxicity[index]) / 255.0 * 0.45, float(biomass[index]) / 255.0 * 0.52, float(moisture[index]) / 255.0 * 0.60, 0.20)
				draw_rect(Rect2(ECOLOGY_ORIGIN + Vector2(x, y) * ECOLOGY_CELL, Vector2(ECOLOGY_CELL, ECOLOGY_CELL)), color, true)
		for organism in organisms: _draw_organism(organism)
	for node in record.get("resource_nodes", []):
		var color: Color = ECOLOGY_COLORS[int(node.get("family_id", 0))]; var position := _grid_to_world(node.get("position", [0, 0]))
		draw_circle(position, 5.5, Color(color.r, color.g, color.b, 0.22)); draw_circle(position, 2.2, color)
	draw_rect(Rect2(ECOLOGY_ORIGIN, Vector2(48, 48) * ECOLOGY_CELL), Color("#2b5970"), false, 1.0)


func _run_ecology_smoke() -> void:
	var errors: Array[String] = startup_errors.duplicate(); var resource_count := 0; var field_cells := 0; var differentiated_metabolism_verified := false; var motive_impulse := 0.0
	var family_census: Dictionary = {}; var behavior_census: Dictionary = {}; var motion_census: Dictionary = {}; var autonomous_motion_verified := true; var organ_capacity_motion_gate_verified := false; var action_cycle_verified := false
	for record in ecology_catalog.get("maps", []):
		resource_count += record.get("resource_nodes", []).size()
		for values in record.get("fields_u8", {}).values(): field_cells += values.size()
	if resource_count != 120: errors.append("resource total")
	if field_cells != 6 * 7 * 48 * 48: errors.append("field cell total")
	if foods.size() < 20: errors.append("live resource population")
	var before := float(foods[0].get("amount", 0.0)); foods[0]["amount"] = maxf(0.0, before - 5.0); _step_food(1.0)
	if float(foods[0].get("amount", 0.0)) <= before - 5.0: errors.append("resource regrowth")
	var animal := {"data": {"family_id": 1}}; var machine := {"data": {"family_id": 4}}; var animal_node := {"family_id": 1}; var machine_node := {"family_id": 4}
	differentiated_metabolism_verified = _resource_affinity(animal, animal_node) > _resource_affinity(animal, machine_node) and _resource_affinity(machine, machine_node) > _resource_affinity(machine, animal_node)
	if not differentiated_metabolism_verified: errors.append("differentiated metabolism")
	var diagnostic_foods: Array = []
	for organism in organisms:
		var family := int(organism["data"].get("family_id", -1)); family_census[str(family)] = int(family_census.get(str(family), 0)) + 1
		autonomous_motion_verified = autonomous_motion_verified and bool(organism.get("motion_autonomous", false))
		var directions := [Vector2.RIGHT, Vector2.DOWN, Vector2.LEFT, Vector2.UP, Vector2(1, 1).normalized()]
		diagnostic_foods.append({"position": _organism_center(organism) + directions[clampi(family, 0, 4)] * 18.0, "amount": 50.0, "capacity": 50.0, "regrowth": 0.0, "ecology_id": 9000 + family, "family_id": family})
	for food in diagnostic_foods: foods.append(food)
	simulation_time = 0.9
	for organism in organisms:
		_step_ecology_behavior(organism, 1.0 / 30.0)
		var motion_name := str(EXPECTED_MOTIONS[int(organism.get("motion_index", 0))]); motion_census[motion_name] = int(motion_census.get(motion_name, 0)) + 1
		var behavior_name := str(organism.get("motion_behavior", "?")); behavior_census[behavior_name] = int(behavior_census.get(behavior_name, 0)) + 1
		motive_impulse += float(organism.get("ecology_motive_impulse", 0.0))
	if family_census.size() != 5 or family_census.values().any(func(value): return int(value) != 1): errors.append("initial family census")
	if not autonomous_motion_verified: errors.append("autonomous motion flags")
	if motion_census.size() < 3 or behavior_census.size() < 5: errors.append("family behavior diversity")
	for organism in organisms:
		if int(organism["data"].get("family_id", -1)) != 1: continue
		var initial_motion := str(EXPECTED_MOTIONS[int(organism.get("motion_index", 0))])
		simulation_time = 1.6; _step_ecology_behavior(organism, 1.0 / 30.0); var recovery_motion := str(EXPECTED_MOTIONS[int(organism.get("motion_index", 0))])
		simulation_time = 3.0; _step_ecology_behavior(organism, 1.0 / 30.0); var replay_motion := str(EXPECTED_MOTIONS[int(organism.get("motion_index", 0))])
		action_cycle_verified = initial_motion == "attack" and recovery_motion.begins_with("idle_") and replay_motion == "attack" and int(organism.get("motion_transition_count", 0)) >= 3
	if not action_cycle_verified: errors.append("action recovery cycle")
	if not organisms.is_empty():
		var diagnostic: Dictionary = organisms[0].duplicate(true); var neural_roles: Array = diagnostic.get("physiology_role", [])[3]
		for index in range(neural_roles.size()):
			if int(neural_roles[index]) == 1: diagnostic["alive"][index] = false; diagnostic["health"][index] = 0.0
		for bond_index in range(diagnostic["bond_ab"].size()):
			var pair: Array = diagnostic["bond_ab"][bond_index]
			if not diagnostic["alive"][int(pair[0])] or not diagnostic["alive"][int(pair[1])]: diagnostic["bond_alive"][bond_index] = false
		diagnostic["physiology_capacities"] = _compute_physiology_capacities(diagnostic)
		_step_ecology_behavior(diagnostic, 1.0 / 30.0)
		organ_capacity_motion_gate_verified = float(diagnostic["physiology_capacities"].get("neural", 1.0)) == 0.0 and str(EXPECTED_MOTIONS[int(diagnostic.get("motion_index", 0))]) == "death"
		if not organ_capacity_motion_gate_verified: errors.append("organ capacity motion gate")
	var report := {"format": "nullvector-cellular-ecology-godot-smoke-v5", "passed": errors.is_empty(), "errors": errors, "engine": Engine.get_version_info().get("string", ""), "ecology_bundle_id": ecology_catalog.get("bundle_id", ""), "motion_bundle_id": motion_catalog.get("bundle_id", ""), "physiology_bundle_id": physiology_catalog.get("bundle_id", ""), "trauma_bundle_id": trauma_catalog.get("bundle_id", ""), "organism_bundle_id": catalog.get("bundle_id", ""), "map_count": ecology_catalog.get("map_count", 0), "resource_node_count": resource_count, "field_cell_count": field_cells, "organism_count": organisms.size(), "family_census": family_census, "motion_census": motion_census, "behavior_census": behavior_census, "autonomous_motion_verified": autonomous_motion_verified, "action_cycle_verified": action_cycle_verified, "organ_capacity_motion_gate_verified": organ_capacity_motion_gate_verified, "ecological_damage_accumulator": ecology_damage_accumulator, "differentiated_metabolism_verified": differentiated_metabolism_verified, "motive_impulse": motive_impulse, "python_runtime_required": false}
	var report_path := "res://../outputs/cellular_ecology_godot_report.json"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--cellular-ecology-report="): report_path = argument.trim_prefix("--cellular-ecology-report=")
	var file := FileAccess.open(report_path, FileAccess.WRITE)
	if file != null: file.store_string(JSON.stringify(report, "  ", false)); file.close()
	print("CELLULAR_ECOLOGY_SMOKE_%s maps=%d resources=%d fields=%d" % ["OK" if errors.is_empty() else "FAIL", int(report["map_count"]), resource_count, field_cells])
	get_tree().quit(0 if errors.is_empty() else 1)
