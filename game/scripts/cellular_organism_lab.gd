extends Node2D

const CATALOG_FORMAT := "nullvector-cellular-organism-native-catalog-v1"
const RUNTIME_FORMAT := "nullvector-cellular-organism-runtime-v1"
const CELL_SCALE := 5.0
const LAB_RECT := Rect2(350.0, 92.0, 900.0, 590.0)
const MAX_ORGANISMS := 8
const MAX_SPILLS := 420
const ORIENTATION_FORMAT := "nullvector-top-down-surface-physics-v1"
const FLAG_EYE := 1
const FLAG_MOUTH := 2
const FLAG_HEART := 4
const FLAG_REPRODUCTIVE := 8
const FLAG_PHOTOSYNTHETIC := 16
const FLAG_WEAPON := 64
const FLAG_EMITTER := 128
const TISSUE_NAMES := ["unused", "epidermis", "contractile", "structural", "neural", "sensory", "vascular", "digestive", "reproductive", "storage", "armor", "weapon", "emitter", "immune", "stem"]
const VIEW_NAMES := ["PHENOTYPE", "ORGANS", "FLUID / PRESSURE", "HEALTH", "TISSUE"]
const CYAN := Color("#39efff")
const PINK := Color("#ff4faf")
const LIME := Color("#b8ff58")
const TEXT := Color("#e9f7ff")
const MUTED := Color("#718ba5")
const RULE := Color("#1d3c5e")
const DEEP := Color("#03070d")

@export_file("*.json") var catalog_path := "res://generated/cellular_organism/v2/catalog.json"
@export_dir var asset_root := "res://generated/cellular_organism/v2/"
@export var expected_species_count := 80
@export var exact_family_count := 16
@export var lineage_mode := false
@export var lab_title := "NULLVECTOR // CELLULAR ORGANISM LAB"
@export var lab_subtitle := "EVERY PIXEL IS A CELL // ORGANS + FLUIDS + FRACTURE + METABOLISM + HEREDITY"

var catalog: Dictionary = {}
var selected_species := 0
var view_mode := 0
var show_bonds := true
var paused := false
var organisms: Array[Dictionary] = []
var foods: Array[Dictionary] = []
var spills: Array[Dictionary] = []
var startup_errors: Array[String] = []
var simulation_time := 0.0
var event_message := "INITIALIZING CELLULAR SUBSTRATE"
var event_time := 0.0
var rng := RandomNumberGenerator.new()

var title_label: Label
var species_label: Label
var anatomy_label: Label
var status_label: Label
var event_label: Label
var view_label: Label
var controls_label: Label


func _ready() -> void:
	get_viewport().set_embedding_subwindows(false)
	rng.seed = 0xC3115EED
	_build_interface()
	catalog = _load_json(catalog_path)
	_validate_catalog()
	if startup_errors.is_empty():
		_spawn_selected(Vector2(790.0, 345.0), 0, 0)
		_refresh_labels()
	else:
		status_label.text = "FAIL-CLOSED // " + ", ".join(startup_errors)
		status_label.modulate = Color("#ff526d")
	queue_redraw()
	if "--cellular-organism-smoke" in OS.get_cmdline_user_args():
		call_deferred("_run_headless_smoke")


func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null: return {}
	var value = JSON.parse_string(file.get_as_text())
	return value if value is Dictionary else {}


func _validate_catalog() -> void:
	if catalog.get("format", "") != CATALOG_FORMAT: startup_errors.append("catalog format")
	if catalog.get("status", "") != "ready": startup_errors.append("catalog status")
	if int(catalog.get("sample_count", -1)) != expected_species_count: startup_errors.append("species census")
	var families: Dictionary = catalog.get("family_counts", {})
	for family in ["humanoid", "animalian", "plantlike", "anomaly", "machine"]:
		var observed := int(families.get(family, -1))
		if exact_family_count > 0 and observed != exact_family_count: startup_errors.append("family " + family)
		if exact_family_count <= 0 and observed < 1: startup_errors.append("family " + family)
	var orientation: Dictionary = catalog.get("orientation", {})
	var surface_contract: Dictionary = orientation.get("contract", {})
	if surface_contract.get("format", "") != ORIENTATION_FORMAT: startup_errors.append("orientation format")
	if surface_contract.get("projection", "") != "top_down_dorsal": startup_errors.append("orientation projection")
	if surface_contract.get("uniform_acceleration_xy", []) != [0.0, 0.0]: startup_errors.append("orientation acceleration")
	if not bool(surface_contract.get("scalar_screen_gravity_disabled", false)): startup_errors.append("orientation gravity gate")
	if surface_contract.get("external_fluid_model", "") != "isotropic_surface_diffusion": startup_errors.append("orientation fluid")
	if str(orientation.get("contract_sha256", "")).length() != 64: startup_errors.append("orientation hash")
	var totals: Dictionary = catalog.get("totals", {})
	var total_cells := int(totals.get("physical_cells", 0))
	if total_cells < expected_species_count * 24: startup_errors.append("cell census")
	if int(totals.get("bonds", 0)) < total_cells - expected_species_count: startup_errors.append("bond census")
	if int(totals.get("organs", 0)) < expected_species_count * 6: startup_errors.append("organ census")
	if catalog.get("species", []).size() != expected_species_count: startup_errors.append("species records")
	if startup_errors.is_empty():
		status_label.text = "%d SPECIES // %s CELLS // %s ORGANS // %s BONDS" % [
			expected_species_count,
			_format_int(int(totals.get("physical_cells", 0))),
			_format_int(int(totals.get("organs", 0))),
			_format_int(int(totals.get("bonds", 0)))
		]
		status_label.modulate = LIME


func _format_int(value: int) -> String:
	var text := str(value)
	var output := ""
	while text.length() > 3:
		output = "," + text.right(3) + output
		text = text.left(text.length() - 3)
	return text + output


func _panel(parent: Node, rect: Rect2) -> Panel:
	var panel := Panel.new()
	panel.position = rect.position; panel.size = rect.size
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.018, 0.045, 0.085, 0.96)
	style.border_color = RULE; style.set_border_width_all(1)
	panel.add_theme_stylebox_override("panel", style); parent.add_child(panel)
	return panel


func _label(parent: Node, position: Vector2, size: Vector2, value: String, color := TEXT, font_size := 11) -> Label:
	var label := Label.new()
	label.position = position; label.size = size; label.text = value
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	parent.add_child(label); return label


func _button(parent: Node, position: Vector2, size: Vector2, value: String, callback: Callable) -> Button:
	var button := Button.new()
	button.position = position; button.size = size; button.text = value; button.focus_mode = Control.FOCUS_NONE
	button.add_theme_font_size_override("font_size", 9); button.pressed.connect(callback); parent.add_child(button)
	return button


func _build_interface() -> void:
	var canvas := CanvasLayer.new(); add_child(canvas)
	title_label = _label(canvas, Vector2(24, 14), Vector2(740, 34), lab_title, TEXT, 23)
	_label(canvas, Vector2(26, 47), Vector2(1050, 20), lab_subtitle, CYAN, 10)
	status_label = _label(canvas, Vector2(700, 19), Vector2(550, 34), "LOADING ANATOMY BANK", LIME, 9)
	status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	var left := _panel(canvas, Rect2(18, 82, 316, 600))
	_label(left, Vector2(14, 10), Vector2(285, 20), "01 // SPECIES / GENOME", CYAN, 11)
	_button(left, Vector2(14, 38), Vector2(45, 30), "Q <", func(): _change_species(-1))
	_button(left, Vector2(63, 38), Vector2(45, 30), "E >", func(): _change_species(1))
	_button(left, Vector2(114, 38), Vector2(86, 30), "RESET X", _reset_species)
	_button(left, Vector2(204, 38), Vector2(95, 30), "VIEW V", _cycle_view)
	species_label = _label(left, Vector2(14, 77), Vector2(285, 76), "SPECIES", PINK, 11)
	species_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_label(left, Vector2(14, 164), Vector2(285, 20), "02 // LIVE ANATOMY", CYAN, 11)
	anatomy_label = _label(left, Vector2(14, 190), Vector2(285, 190), "ANATOMY", TEXT, 9)
	anatomy_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_label(left, Vector2(14, 389), Vector2(285, 20), "03 // INTERVENTIONS", CYAN, 11)
	_button(left, Vector2(14, 417), Vector2(86, 31), "FEED F", func(): _feed_current(35.0))
	_button(left, Vector2(105, 417), Vector2(92, 31), "REPRO R", _induce_reproduction)
	_button(left, Vector2(202, 417), Vector2(97, 31), "BLAST SPC", _blast_current)
	_button(left, Vector2(14, 454), Vector2(86, 31), "BONDS B", func(): show_bonds = not show_bonds; queue_redraw())
	var orientation_label := _label(left, Vector2(105, 454), Vector2(92, 31), "TOP-DOWN\nSURFACE XY", CYAN, 8)
	orientation_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER; orientation_label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	_button(left, Vector2(202, 454), Vector2(97, 31), "PAUSE P", func(): paused = not paused)
	controls_label = _label(left, Vector2(14, 500), Vector2(285, 82), "LMB / DRAG  tear tissue\nRMB  place food\nPuddles diffuse across surface XY\nV  phenotype / organs / fluid / health / tissue", MUTED, 9)
	controls_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	view_label = _label(canvas, Vector2(365, 102), Vector2(400, 24), "VIEW // PHENOTYPE", CYAN, 10)
	event_label = _label(canvas, Vector2(365, 650), Vector2(870, 25), event_message, PINK, 10)


func _runtime_entry(index: int) -> Dictionary:
	return catalog.get("species", [])[posmod(index, catalog.get("species", []).size())]


func _load_species_data(index: int) -> Dictionary:
	var entry := _runtime_entry(index)
	var runtime: Dictionary = entry.get("runtime", {})
	var path := asset_root + str(runtime.get("path", ""))
	if not FileAccess.file_exists(path): return {}
	if FileAccess.get_file_as_bytes(path).size() != int(runtime.get("bytes", -1)): return {}
	if FileAccess.get_sha256(path) != str(runtime.get("sha256", "")): return {}
	var data := _load_json(path)
	if data.get("format", "") != RUNTIME_FORMAT: return {}
	return data


func _spawn_selected(center: Vector2, generation: int, mutation_seed: int) -> Dictionary:
	var data := _load_species_data(selected_species)
	if data.is_empty():
		startup_errors.append("runtime species")
		return {}
	var organism := _create_organism(data, center, generation, mutation_seed)
	organisms.append(organism)
	_event("BIRTH // %s // GENERATION %d" % [str(data.get("sample_id", "?")), generation], LIME)
	return organism


func _create_organism(data: Dictionary, center: Vector2, generation: int, mutation_seed: int) -> Dictionary:
	var arrays: Dictionary = data.get("arrays", {})
	var positions: Array = []
	var centroid := Vector2.ZERO
	for item in arrays.get("position_xy", []):
		var point := Vector2(float(item[0]), float(item[1])) * CELL_SCALE
		positions.append(point); centroid += point
	centroid /= max(1, positions.size())
	for index in range(positions.size()): positions[index] = positions[index] - centroid + center
	var count := positions.size()
	var zeros: Array = []; var alive: Array = []; var bond_alive: Array = []
	for index in range(count): zeros.append(Vector2.ZERO); alive.append(true)
	for index in range(arrays.get("bond_ab", []).size()): bond_alive.append(true)
	var genome: Dictionary = data.get("genome", {}).duplicate(true)
	genome["generation"] = generation
	if mutation_seed != 0:
		genome["genome_seed"] = mutation_seed
		_mutate_genome(genome, mutation_seed)
	var open_bonds: Array = []; var incident: Array = []
	for index in range(count): open_bonds.append(0); incident.append(0)
	for pair in arrays.get("bond_ab", []):
		incident[int(pair[0])] += 1; incident[int(pair[1])] += 1
	return {
		"data": data,
		"genome": genome,
		"position": positions,
		"velocity": zeros,
		"health": arrays.get("max_health", []).duplicate(),
		"max_health": arrays.get("max_health", []).duplicate(),
		"fluid": arrays.get("fluid_initial", []).duplicate(),
		"fluid_baseline": arrays.get("fluid_initial", []).duplicate(),
		"fluid_capacity": arrays.get("fluid_capacity", []).duplicate(),
		"nutrient": arrays.get("nutrient_initial", []).duplicate(),
		"energy": arrays.get("energy_initial", []).duplicate(),
		"mass": arrays.get("mass", []).duplicate(),
		"tissue": arrays.get("tissue", []).duplicate(),
		"organ_id": arrays.get("organ_id", []).duplicate(),
		"flags": arrays.get("cell_flags", []).duplicate(),
		"material": arrays.get("material", []).duplicate(),
		"emission": arrays.get("emission", []).duplicate(),
		"bond_ab": arrays.get("bond_ab", []).duplicate(true),
		"bond_rest": arrays.get("bond_rest", []).duplicate(),
		"bond_strength": arrays.get("bond_strength", []).duplicate(),
		"bond_conductance": arrays.get("bond_conductance", []).duplicate(),
		"bond_alive": bond_alive,
		"alive": alive,
		"incident": incident,
		"open_bonds": open_bonds,
		"age": 0.0,
		"birth_cooldown": float(genome.get("gestation_seconds", 10.0)),
		"fluid_lost": 0.0,
		"food_consumed": 0.0,
		"damage_taken": 0.0,
		"births": 0,
		"palette_shift": fmod(float(mutation_seed % 997) / 997.0 * 0.16, 1.0) if mutation_seed != 0 else 0.0,
	}


func _mutate_genome(genome: Dictionary, seed_value: int) -> void:
	var local := RandomNumberGenerator.new(); local.seed = seed_value
	for trait_name in ["metabolic_rate", "digestion_efficiency", "fluid_regeneration_rate", "tissue_regeneration_rate", "reproduction_energy_threshold"]:
		if local.randf() < float(genome.get("mutation_rate", 0.04)):
			genome[trait_name] = float(genome.get(trait_name, 1.0)) * (1.0 + local.randf_range(-1.0, 1.0) * float(genome.get("mutation_scale", 0.08)))


func _physics_process(delta: float) -> void:
	if paused or not startup_errors.is_empty(): return
	var bounded_delta: float = minf(delta, 1.0 / 30.0)
	for substep in range(2):
		var sub_delta: float = bounded_delta * 0.5
		for organism in organisms: _step_organism(organism, sub_delta)
		_step_food(sub_delta); _step_spills(sub_delta)
	simulation_time += bounded_delta
	event_time = max(0.0, event_time - bounded_delta)
	_try_natural_reproduction()
	_refresh_labels(); queue_redraw()


func _step_organism(organism: Dictionary, delta: float) -> void:
	var positions: Array = organism["position"]; var velocity: Array = organism["velocity"]
	var alive: Array = organism["alive"]; var pairs: Array = organism["bond_ab"]; var bond_alive: Array = organism["bond_alive"]
	var rest: Array = organism["bond_rest"]; var strength: Array = organism["bond_strength"]; var mass: Array = organism["mass"]
	for bond_index in range(pairs.size()):
		if not bond_alive[bond_index]: continue
		var a := int(pairs[bond_index][0]); var b := int(pairs[bond_index][1])
		if not alive[a] or not alive[b]: _break_bond(organism, bond_index); continue
		var difference: Vector2 = positions[b] - positions[a]
		var length: float = maxf(0.001, difference.length()); var target: float = float(rest[bond_index]) * CELL_SCALE
		var scar_fraction := 0.0
		if organism.has("trauma_scar") and organism["trauma_scar"].size() == positions.size(): scar_fraction = (float(organism["trauma_scar"][a]) + float(organism["trauma_scar"][b])) * 0.5
		if length / target > 1.25 + float(strength[bond_index]) * 0.22 - scar_fraction * 0.08:
			_break_bond(organism, bond_index); continue
		var scar_stiffness := 1.0 + scar_fraction * 0.45
		var force: Vector2 = difference / length * (length - target) * float(strength[bond_index]) * scar_stiffness * 7.5 * delta
		velocity[a] += force / max(0.1, float(mass[a])); velocity[b] -= force / max(0.1, float(mass[b]))
	var living_center := _organism_center(organism)
	for cell_index in range(positions.size()):
		if not alive[cell_index]:
			var toward: Vector2 = living_center - positions[cell_index]
			var distance: float = maxf(18.0, toward.length())
			velocity[cell_index] += toward * minf(0.0018, 0.055 / distance) * delta
		velocity[cell_index] *= 0.985; positions[cell_index] += velocity[cell_index] * delta
		if positions[cell_index].x < LAB_RECT.position.x:
			positions[cell_index].x = LAB_RECT.position.x; velocity[cell_index].x = abs(velocity[cell_index].x) * 0.15
		if positions[cell_index].x > LAB_RECT.end.x:
			positions[cell_index].x = LAB_RECT.end.x; velocity[cell_index].x = -abs(velocity[cell_index].x) * 0.15
		if positions[cell_index].y < LAB_RECT.position.y:
			positions[cell_index].y = LAB_RECT.position.y; velocity[cell_index].y = abs(velocity[cell_index].y) * 0.15
		if positions[cell_index].y > LAB_RECT.end.y:
			positions[cell_index].y = LAB_RECT.end.y; velocity[cell_index].y = -abs(velocity[cell_index].y) * 0.15
	_step_fluid_and_metabolism(organism, delta)
	organism["age"] = float(organism["age"]) + delta
	organism["birth_cooldown"] = max(0.0, float(organism["birth_cooldown"]) - delta)


func _break_bond(organism: Dictionary, bond_index: int) -> void:
	if not organism["bond_alive"][bond_index]: return
	organism["bond_alive"][bond_index] = false
	var pair: Array = organism["bond_ab"][bond_index]
	organism["open_bonds"][int(pair[0])] += 1; organism["open_bonds"][int(pair[1])] += 1


func _step_fluid_and_metabolism(organism: Dictionary, delta: float) -> void:
	var fluid: Array = organism["fluid"]; var capacity: Array = organism["fluid_capacity"]
	var pairs: Array = organism["bond_ab"]; var conductance: Array = organism["bond_conductance"]
	var bond_alive: Array = organism["bond_alive"]; var alive: Array = organism["alive"]
	var fluid_delta: Array = []; for index in range(fluid.size()): fluid_delta.append(0.0)
	for bond_index in range(pairs.size()):
		if not bond_alive[bond_index]: continue
		var a := int(pairs[bond_index][0]); var b := int(pairs[bond_index][1])
		if not alive[a] or not alive[b]: continue
		var pressure_a: float = float(fluid[a]) / maxf(0.001, float(capacity[a]))
		var pressure_b: float = float(fluid[b]) / maxf(0.001, float(capacity[b]))
		var transfer: float = clampf((pressure_a - pressure_b) * float(conductance[bond_index]) * 2.2 * delta, -float(fluid[b]), float(fluid[a]))
		fluid_delta[a] -= transfer; fluid_delta[b] += transfer
	var health: Array = organism["health"]; var max_health: Array = organism["max_health"]
	var energy: Array = organism["energy"]; var nutrient: Array = organism["nutrient"]; var tissue: Array = organism["tissue"]
	var flags: Array = organism["flags"]; var metabolic: float = float(organism["genome"].get("metabolic_rate", 1.0)) / maxi(1, fluid.size())
	for index in range(fluid.size()):
		fluid[index] = max(0.0, float(fluid[index]) + float(fluid_delta[index]))
		var exposure: float = float(organism["open_bonds"][index]) / maxf(1.0, float(organism["incident"][index]))
		exposure += clamp(1.0 - float(health[index]) / max(0.001, float(max_health[index])), 0.0, 1.0)
		if not alive[index]: exposure += 2.0
		var clot_fraction := 0.0
		if organism.has("trauma_clot") and organism["trauma_clot"].size() == fluid.size(): clot_fraction = clampf(float(organism["trauma_clot"][index]), 0.0, 1.0)
		var leaked: float = minf(float(fluid[index]), exposure * (1.0 - clot_fraction) * 0.7 * delta)
		if leaked > 0.0005:
			fluid[index] -= leaked; organism["fluid_lost"] = float(organism["fluid_lost"]) + leaked
			if rng.randf() < min(1.0, leaked * 10.0): _spawn_spill(organism["position"][index], organism, leaked)
		if alive[index] and int(tissue[index]) == 7 and float(nutrient[index]) > 0.0:
			var converted: float = minf(float(nutrient[index]), float(organism["genome"].get("digestion_efficiency", 0.7)) * delta)
			nutrient[index] -= converted; energy[index] = float(energy[index]) + converted * 8.0
		if alive[index] and (int(flags[index]) & FLAG_PHOTOSYNTHETIC) != 0: energy[index] = float(energy[index]) + 0.012 * delta
		if alive[index]:
			energy[index] = max(0.0, float(energy[index]) - metabolic * delta)
			if float(energy[index]) <= 0.00001:
				health[index] = float(health[index]) - 0.025 * delta
				if float(health[index]) <= 0.0: alive[index] = false
			elif int(tissue[index]) == 6 and float(health[index]) < float(max_health[index]):
				var healing: float = minf(float(max_health[index]) - float(health[index]), float(organism["genome"].get("tissue_regeneration_rate", 0.01)) * delta)
				health[index] += healing; energy[index] = max(0.0, float(energy[index]) - healing * 0.45)


func _step_food(delta: float) -> void:
	for food in foods:
		food["pulse"] = fmod(float(food["pulse"]) + delta * 3.0, TAU)
	for organism in organisms:
		for cell_index in range(organism["position"].size()):
			if not organism["alive"][cell_index] or (int(organism["flags"][cell_index]) & FLAG_MOUTH) == 0: continue
			for food in foods:
				if float(food["amount"]) <= 0.0: continue
				if organism["position"][cell_index].distance_to(food["position"]) < 13.0:
					var bite: float = minf(float(food["amount"]), 8.0 * delta)
					food["amount"] -= bite; organism["nutrient"][cell_index] += bite; organism["food_consumed"] += bite
	foods = foods.filter(func(food: Dictionary): return float(food["amount"]) > 0.01)


func _step_spills(delta: float) -> void:
	for spill in spills:
		var age: float = float(spill["age"]) + delta; spill["age"] = age
		var curl: float = sin(float(spill["seed"]) * 37.0 + age * 1.1) * delta * 0.7
		spill["velocity"] = Vector2(spill["velocity"]).rotated(curl) * pow(0.958, delta * 60.0)
		var spill_position: Vector2 = Vector2(spill["position"]) + Vector2(spill["velocity"]) * delta
		spill_position.x = clampf(spill_position.x, LAB_RECT.position.x, LAB_RECT.end.x)
		spill_position.y = clampf(spill_position.y, LAB_RECT.position.y, LAB_RECT.end.y)
		spill["position"] = spill_position
		spill["radius"] = minf(22.0, float(spill["radius"]) + delta * (4.2 + minf(8.0, float(spill["amount"]) * 55.0)))
		spill["life"] = float(spill["life"]) - delta * (0.085 + minf(0.055, float(spill["amount"]) * 0.4))
	spills = spills.filter(func(spill: Dictionary): return float(spill["life"]) > 0.0)


func _spawn_spill(position: Vector2, organism: Dictionary, amount: float) -> void:
	if spills.size() >= MAX_SPILLS: return
	var palette: Dictionary = organism["data"].get("palette", {})
	var rgb: Array = palette.get("fluid_rgb", [255, 50, 100])
	var center := _organism_center(organism)
	var radial := (position - center).normalized()
	var seed := rng.randf()
	if radial.length_squared() < 0.01: radial = Vector2.RIGHT.rotated(seed * TAU)
	var angle_jitter := sin(seed * 31.7 + simulation_time * 0.66) * 0.72
	spills.append({
		"position": position,
		"velocity": radial.rotated(angle_jitter) * (10.0 + minf(38.0, amount * 220.0)),
		"life": 1.0,
		"age": 0.0,
		"radius": 1.5 + minf(2.5, amount * 18.0),
		"amount": amount,
		"seed": seed,
		"color": Color8(int(rgb[0]), int(rgb[1]), int(rgb[2]), 210),
	})


func _draw_surface_spill(spill: Dictionary) -> void:
	var density: float = maxf(0.0, float(spill["life"])); var radius: float = maxf(2.0, float(spill["radius"]))
	var center: Vector2 = Vector2(spill["position"]).round(); var base: Color = spill["color"]
	var core := base; core.a = density * 0.28
	draw_rect(Rect2(center - Vector2.ONE * radius * 0.28, Vector2.ONE * radius * 0.56), core)
	for lobe in range(12):
		var angle: float = float(spill["seed"]) * TAU + float(lobe) * PI / 6.0
		var phase: float = float((lobe * 7) % 11) / 10.0
		var distance: float = radius * (0.3 + phase * 0.62); var size: float = maxf(1.5, 4.2 - phase * 2.4)
		var point := (center + Vector2.RIGHT.rotated(angle) * distance).round()
		var color := base; color.a = density * (0.36 - phase * 0.18)
		draw_rect(Rect2(point - Vector2.ONE * size * 0.5, Vector2.ONE * size), color)


func _damage_at(point: Vector2, radius := 24.0, damage := 1.45, impulse := 135.0) -> Dictionary:
	var totals := {"cells": 0, "killed": 0, "bonds": 0}
	for organism in organisms:
		var positions: Array = organism["position"]; var health: Array = organism["health"]; var alive: Array = organism["alive"]
		var velocity: Array = organism["velocity"]; var mass: Array = organism["mass"]
		var affected: Array = []
		for index in range(positions.size()):
			if not alive[index]: continue
			var distance: float = positions[index].distance_to(point)
			if distance >= radius: continue
			var falloff := 1.0 - distance / radius; affected.append(index); totals["cells"] += 1
			health[index] = float(health[index]) - damage * falloff; organism["damage_taken"] += damage * falloff
			var direction: Vector2 = (positions[index] - point).normalized() if distance > 0.1 else Vector2.UP.rotated(rng.randf() * TAU)
			velocity[index] += direction * impulse * falloff / max(0.1, float(mass[index]))
			if float(health[index]) <= 0.0:
				alive[index] = false; health[index] = 0.0; totals["killed"] += 1
				_spawn_spill(positions[index], organism, 0.25)
		for bond_index in range(organism["bond_ab"].size()):
			if not organism["bond_alive"][bond_index]: continue
			var pair: Array = organism["bond_ab"][bond_index]
			if not organism["alive"][int(pair[0])] or not organism["alive"][int(pair[1])]:
				_break_bond(organism, bond_index); totals["bonds"] += 1; continue
			if int(pair[0]) in affected or int(pair[1]) in affected:
				var midpoint: Vector2 = (positions[int(pair[0])] + positions[int(pair[1])]) * 0.5
				var local_falloff: float = maxf(0.0, 1.0 - midpoint.distance_to(point) / radius)
				if impulse * local_falloff * 0.018 > float(organism["bond_strength"][bond_index]):
					_break_bond(organism, bond_index); totals["bonds"] += 1
	if int(totals["cells"]) > 0: _event("TRAUMA // %d CELLS // %d KILLED // %d BONDS TORN" % [totals["cells"], totals["killed"], totals["bonds"]], PINK)
	return totals


func _feed_current(amount: float) -> float:
	if organisms.is_empty(): return 0.0
	var organism := organisms[0]; var recipients: Array = []
	for index in range(organism["flags"].size()):
		if organism["alive"][index] and ((int(organism["flags"][index]) & FLAG_MOUTH) != 0 or int(organism["tissue"][index]) == 7): recipients.append(index)
	if recipients.is_empty(): _event("FEED FAILED // DIGESTIVE ORGAN LOST", Color("#ff526d")); return 0.0
	for index in recipients: organism["nutrient"][index] += amount / recipients.size()
	organism["food_consumed"] += amount
	_event("FED // +%.1f NUTRIENTS" % amount, LIME)
	return amount


func _can_reproduce(organism: Dictionary) -> bool:
	var has_brain := false; var has_heart := false; var has_repro := false; var alive_count := 0; var total_energy := 0.0
	for index in range(organism["alive"].size()):
		if not organism["alive"][index]: continue
		alive_count += 1; total_energy += float(organism["energy"][index])
		if int(organism["tissue"][index]) == 4: has_brain = true
		if (int(organism["flags"][index]) & FLAG_HEART) != 0: has_heart = true
		if (int(organism["flags"][index]) & FLAG_REPRODUCTIVE) != 0: has_repro = true
	return has_brain and has_heart and has_repro and float(alive_count) / max(1, organism["alive"].size()) >= 0.75 and total_energy >= float(organism["genome"].get("reproduction_energy_threshold", 100.0)) and float(organism["birth_cooldown"]) <= 0.0


func _reproduce(organism: Dictionary, forced := false) -> bool:
	if organisms.size() >= MAX_ORGANISMS: return false
	if forced:
		var threshold := float(organism["genome"].get("reproduction_energy_threshold", 100.0))
		var current := _sum_float(organism["energy"])
		if current < threshold:
			var add: float = (threshold - current + 4.0) / maxi(1, organism["energy"].size())
			for index in range(organism["energy"].size()): organism["energy"][index] += add
		organism["birth_cooldown"] = 0.0
	if not _can_reproduce(organism):
		_event("REPRODUCTION BLOCKED // ENERGY OR ESSENTIAL ORGAN FAILURE", Color("#ff526d")); return false
	var fraction := float(organism["genome"].get("offspring_energy_fraction", 0.35))
	for index in range(organism["energy"].size()): organism["energy"][index] *= 1.0 - fraction
	var seed_value := int(organism["genome"].get("genome_seed", 1)) ^ (organisms.size() * 0x9E3779B1) ^ int(simulation_time * 1000.0 + 17.0)
	var center := _organism_center(organism) + Vector2(140.0 if organisms.size() % 2 else -140.0, 65.0)
	var child := _create_organism(organism["data"], center, int(organism["genome"].get("generation", 0)) + 1, seed_value)
	for index in range(child["energy"].size()): child["energy"][index] *= fraction
	organisms.append(child); organism["births"] += 1; organism["birth_cooldown"] = float(organism["genome"].get("gestation_seconds", 10.0))
	_event("REPRODUCTION // GENERATION %d // MUTATION SEED %08X" % [child["genome"]["generation"], seed_value & 0xFFFFFFFF], LIME)
	return true


func _try_natural_reproduction() -> void:
	if organisms.size() >= MAX_ORGANISMS: return
	for organism in organisms.duplicate():
		if _can_reproduce(organism): _reproduce(organism); return


func _induce_reproduction() -> void:
	if not organisms.is_empty(): _reproduce(organisms[0], true)


func _blast_current() -> void:
	if organisms.is_empty(): return
	_damage_at(_organism_center(organisms[0]), 44.0, 0.85, 260.0)


func _organism_center(organism: Dictionary) -> Vector2:
	var center := Vector2.ZERO; var count := 0
	for index in range(organism["position"].size()):
		if organism["alive"][index]: center += organism["position"][index]; count += 1
	return center / max(1, count)


func _sum_float(values: Array) -> float:
	var total := 0.0
	for value in values: total += float(value)
	return total


func _change_species(delta: int) -> void:
	if catalog.get("species", []).is_empty(): return
	selected_species = posmod(selected_species + delta, catalog.get("species", []).size())
	_reset_species()


func _reset_species() -> void:
	organisms.clear(); foods.clear(); spills.clear(); simulation_time = 0.0
	_spawn_selected(Vector2(790.0, 345.0), 0, 0); _refresh_labels(); queue_redraw()


func _cycle_view() -> void:
	view_mode = (view_mode + 1) % VIEW_NAMES.size(); view_label.text = "VIEW // " + VIEW_NAMES[view_mode]; queue_redraw()


func _event(message: String, color: Color) -> void:
	event_message = message; event_time = 2.5
	if event_label != null: event_label.text = message; event_label.modulate = color


func _refresh_labels() -> void:
	if catalog.is_empty() or organisms.is_empty(): return
	var entry := _runtime_entry(selected_species); var organism := organisms[0]; var data: Dictionary = organism["data"]
	species_label.text = "%02d // %s\n%s / %s\n%s // %s" % [selected_species + 1, str(entry.get("sample_id", "?")), str(entry.get("family", "?")).to_upper(), str(entry.get("subtype", "?")), str(entry.get("role", "?")).to_upper(), str(data.get("fluid", {}).get("name", "fluid")).to_upper()]
	if lineage_mode:
		var lineage: Dictionary = data.get("lineage", {})
		species_label.text += "\nG%d R%02d // %s + %s\n%s // %s" % [int(lineage.get("generation", 0)), int(lineage.get("rank", -1)), str(lineage.get("parent_ids", ["?", "?"])[0]), str(lineage.get("parent_ids", ["?", "?"])[1]), str(lineage.get("fusion_mode", "?")).to_upper(), str(lineage.get("mutation_mode", "?")).to_upper()]
	var alive_count := 0; var intact := 0
	for value in organism["alive"]: if value: alive_count += 1
	for value in organism["bond_alive"]: if value: intact += 1
	var health: float = _sum_float(organism["health"]) / maxf(0.001, _sum_float(organism["max_health"]))
	var fluid: float = _sum_float(organism["fluid"]) / maxf(0.001, _sum_float(organism["fluid_capacity"]))
	var summary: Dictionary = data.get("summary", {})
	anatomy_label.text = "GENERATION  %d   POPULATION  %d\nCELLS  %d / %d   HEALTH  %5.1f%%\nBONDS  %d / %d   FLUID  %5.1f%%\nORGANS  %d   EYES  %d   APPENDAGES  %d\nENERGY  %7.2f   NUTRIENTS  %7.2f\nFOOD EATEN  %6.2f   FLUID LOST  %6.2f\nDIET  %s\nREPRODUCTION  %s\nREADY  %s" % [int(organism["genome"].get("generation", 0)), organisms.size(), alive_count, organism["alive"].size(), health * 100.0, intact, organism["bond_alive"].size(), fluid * 100.0, int(summary.get("organ_count", 0)), int(summary.get("eye_count", 0)), int(summary.get("appendage_organ_count", 0)), _sum_float(organism["energy"]), _sum_float(organism["nutrient"]), float(organism["food_consumed"]), float(organism["fluid_lost"]), str(organism["genome"].get("diet", "?")), str(organism["genome"].get("reproduction_mode", "?")), "YES" if _can_reproduce(organism) else "NO"]
	if event_time <= 0.0: event_label.modulate = MUTED; event_label.text = "LIVE // LMB TEAR // RMB FOOD // F FEED // R REPRODUCE // SPACE BLAST"


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_Q: _change_species(-1)
			KEY_E: _change_species(1)
			KEY_V: _cycle_view()
			KEY_B: show_bonds = not show_bonds; queue_redraw()
			KEY_P: paused = not paused
			KEY_F: _feed_current(35.0)
			KEY_R: _induce_reproduction()
			KEY_X: _reset_species()
			KEY_SPACE: _blast_current()
	if event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_LEFT: _damage_at(event.position)
		if event.button_index == MOUSE_BUTTON_RIGHT:
			foods.append({"position": event.position, "amount": 30.0, "pulse": 0.0}); _event("FOOD PLACED // MOUTH CELLS WILL INGEST", LIME)
	if event is InputEventMouseMotion and (event.button_mask & MOUSE_BUTTON_MASK_LEFT) != 0:
		_damage_at(event.position, 15.0, 0.22, 65.0)


func _draw() -> void:
	draw_rect(Rect2(0, 0, 1280, 720), DEEP)
	draw_rect(LAB_RECT, Color("#050d17"), true)
	for x in range(int(LAB_RECT.position.x), int(LAB_RECT.end.x), 24): draw_line(Vector2(x, LAB_RECT.position.y), Vector2(x, LAB_RECT.end.y), Color(0.08, 0.18, 0.25, 0.35), 1)
	for y in range(int(LAB_RECT.position.y), int(LAB_RECT.end.y), 24): draw_line(Vector2(LAB_RECT.position.x, y), Vector2(LAB_RECT.end.x, y), Color(0.08, 0.18, 0.25, 0.35), 1)
	for food in foods:
		var radius := 5.0 + sin(float(food["pulse"])) * 1.5
		draw_circle(food["position"], radius + 4.0, Color(0.7, 1.0, 0.25, 0.1)); draw_circle(food["position"], radius, LIME)
	for spill in spills: _draw_surface_spill(spill)
	for organism in organisms: _draw_organism(organism)
	draw_rect(LAB_RECT, RULE, false, 1)


func _draw_organism(organism: Dictionary) -> void:
	var positions: Array = organism["position"]; var alive: Array = organism["alive"]
	if show_bonds:
		for bond_index in range(organism["bond_ab"].size()):
			if not organism["bond_alive"][bond_index]: continue
			var pair: Array = organism["bond_ab"][bond_index]; var a := int(pair[0]); var b := int(pair[1])
			var bond_color := Color(0.2, 0.75, 0.9, 0.09) if alive[a] and alive[b] else Color(1, 0.15, 0.3, 0.08)
			draw_line(positions[a], positions[b], bond_color, 1.0)
	for index in range(positions.size()):
		var color := _cell_color(organism, index)
		if not alive[index]: color = Color(color.r * 0.35, color.g * 0.25, color.b * 0.25, 0.55)
		if alive[index] and int(organism["emission"][index]) > 0:
			draw_circle(positions[index], 4.0 + int(organism["emission"][index]), Color(color.r, color.g, color.b, 0.08))
		draw_rect(Rect2(positions[index] - Vector2(2.15, 2.15), Vector2(4.3, 4.3)), color, true)
		if alive[index] and (int(organism["flags"][index]) & FLAG_EYE) != 0:
			draw_rect(Rect2(positions[index] - Vector2(1.2, 1.2), Vector2(2.4, 2.4)), Color.WHITE, true)


func _cell_color(organism: Dictionary, index: int) -> Color:
	var data: Dictionary = organism["data"]; var palette: Dictionary = data.get("palette", {})
	if view_mode == 0:
		var colors: Array = palette.get("material_mid_rgb", [])
		var rgb: Array = colors[int(organism["material"][index])] if int(organism["material"][index]) < colors.size() else [80, 180, 210]
		var color := Color8(int(rgb[0]), int(rgb[1]), int(rgb[2]))
		if float(organism["palette_shift"]) != 0.0:
			color = Color.from_hsv(fmod(color.h + float(organism["palette_shift"]), 1.0), color.s, color.v)
		return color
	if view_mode == 1:
		var organ := int(organism["organ_id"][index]); return Color.from_hsv(fmod(float(organ) * 0.61803398875, 1.0), 0.72, 0.92)
	if view_mode == 2:
		var fluid_rgb: Array = palette.get("fluid_rgb", [255, 60, 110]); var ratio: float = float(organism["fluid"][index]) / maxf(0.001, float(organism["fluid_capacity"][index]))
		return Color(float(fluid_rgb[0]) / 255.0 * ratio, float(fluid_rgb[1]) / 255.0 * ratio, float(fluid_rgb[2]) / 255.0 * ratio, 1.0)
	if view_mode == 3:
		var ratio: float = float(organism["health"][index]) / maxf(0.001, float(organism["max_health"][index])); return Color(1.0 - ratio, ratio, 0.2, 1.0)
	var tissue_colors := [Color.BLACK, Color("#4bc9d8"), Color("#ef4d78"), Color("#9aa7b5"), Color("#956cff"), Color("#f7ec70"), Color("#ff3d67"), Color("#ff9e3d"), Color("#ff60bd"), Color("#62db8b"), Color("#6f7e98"), Color("#ff653e"), Color("#3bf7ff"), Color("#d9fbef"), Color("#8cff4f")]
	return tissue_colors[int(organism["tissue"][index])]


func _run_headless_smoke() -> void:
	var report_path := ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--cellular-organism-report="): report_path = argument.trim_prefix("--cellular-organism-report=")
	var errors: Array[String] = startup_errors.duplicate()
	var total_cells := 0; var total_bonds := 0; var total_organs := 0; var loaded := 0
	for species_index in range(catalog.get("species", []).size()):
		var data := _load_species_data(species_index)
		if data.is_empty(): errors.append("load species %d" % species_index); continue
		var arrays: Dictionary = data.get("arrays", {}); var summary: Dictionary = data.get("summary", {})
		var cells: int = arrays.get("position_xy", []).size(); var bonds: int = arrays.get("bond_ab", []).size()
		if cells != int(summary.get("physical_cell_count", -1)): errors.append("cell count %d" % species_index)
		if bonds != int(summary.get("bond_count", -1)): errors.append("bond count %d" % species_index)
		if arrays.get("tissue", []).size() != cells or arrays.get("fluid_initial", []).size() != cells: errors.append("cell arrays %d" % species_index)
		total_cells += cells; total_bonds += bonds; total_organs += data.get("organs", []).size(); loaded += 1
	var before_bonds := 0; var before_fluid := 0.0
	if not organisms.is_empty():
		before_bonds = organisms[0]["bond_alive"].size(); before_fluid = _sum_float(organisms[0]["fluid"])
		var trauma := _damage_at(_organism_center(organisms[0]), 38.0, 2.2, 260.0)
		if int(trauma["killed"]) <= 0 or int(trauma["bonds"]) <= 0: errors.append("damage mechanics")
		for step in range(20): _step_organism(organisms[0], 1.0 / 120.0)
		if _sum_float(organisms[0]["fluid"]) >= before_fluid: errors.append("fluid leakage")
		if _feed_current(20.0) <= 0.0: errors.append("feeding")
		if not _reproduce(organisms[0], true): errors.append("reproduction")
	if total_cells != int(catalog.get("totals", {}).get("physical_cells", -1)): errors.append("total cells")
	if total_bonds != int(catalog.get("totals", {}).get("bonds", -1)): errors.append("total bonds")
	if total_organs != int(catalog.get("totals", {}).get("organs", -1)): errors.append("total organs")
	var report := {
		"passed": errors.is_empty(), "errors": errors, "engine": Engine.get_version_info().get("string", ""),
		"catalog_bundle_id": catalog.get("bundle_id", ""), "species_loaded": loaded,
		"cells_checked": total_cells, "organs_checked": total_organs, "bonds_checked": total_bonds,
		"damage_bond_baseline": before_bonds, "fluid_baseline": before_fluid,
		"population_after_reproduction": organisms.size(), "python_runtime_required": false,
		"orientation": "top_down_dorsal", "uniform_screen_gravity": false,
		"surface_fluid_model": "isotropic_surface_diffusion", "surface_puddles_observed": spills.size(),
	}
	if not report_path.is_empty():
		var file := FileAccess.open(report_path, FileAccess.WRITE)
		if file != null: file.store_string(JSON.stringify(report, "  ", true) + "\n")
	if errors.is_empty(): print("CELLULAR_ORGANISM_SMOKE_OK species=%d cells=%d organs=%d bonds=%d population=%d" % [expected_species_count, total_cells, total_organs, total_bonds, organisms.size()])
	else: push_error("CELLULAR_ORGANISM_SMOKE_FAIL " + ", ".join(errors))
	get_tree().quit(0 if errors.is_empty() else 1)
