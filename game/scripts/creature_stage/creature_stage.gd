extends Node2D

const Neural = preload("res://scripts/creature_stage/creature_neural.gd")
const Creature = preload("res://scripts/creature_stage/neural_creature.gd")
const World = preload("res://scripts/creature_stage/neural_world.gd")

const WORLD_SEED := 0x4E554C4C
const MAX_ACTIVE_CREATURES := 86
const VIEW_SIZE := Vector2(1280, 720)
const TRACE_STEPS := 240
const TRACE_DELTA := 1.0 / 30.0
const TRAITS := {
	"paired_graspers": {"name": "Paired Graspers", "text": "Manipulation and construction speed +35%."},
	"redundant_pulse": {"name": "Redundant Pulse", "text": "Circulation remains functional after severe heart loss."},
	"echo_memory": {"name": "Echo Memory", "text": "Sensory targets persist outside the current cone."},
	"pack_resonance": {"name": "Pack Resonance", "text": "Nearby kin coordinate hunting and flight."},
	"venom_gland": {"name": "Venom Gland", "text": "Close attacks inflict delayed tissue failure."},
	"photosymbiotic": {"name": "Photosymbiotic", "text": "Slow energy generation in nutrient-rich light fields."},
	"tessellating_runners": {"name": "Tessellating Runners", "text": "Plant offspring can bud from remote root tips."},
	"phase_mouth": {"name": "Phase Mouth", "text": "Consume anomaly flux and some incoming projectiles."},
	"nonlocal_lobe": {"name": "Nonlocal Lobe", "text": "Utility action blinks through ordinary obstacles."},
	"rail_hardpoints": {"name": "Rail Hardpoints", "text": "Paired high-velocity mineral projectiles."},
	"coolant_recycler": {"name": "Coolant Recycler", "text": "Reduced hydration loss and improved repair."},
	"lithovore": {"name": "Lithovore", "text": "Digest mineral resources as energy and construction mass."},
}
const FAMILY_TRAITS := [
	["paired_graspers", "redundant_pulse", "echo_memory"],
	["pack_resonance", "venom_gland", "echo_memory"],
	["photosymbiotic", "tessellating_runners", "redundant_pulse"],
	["phase_mouth", "nonlocal_lobe", "echo_memory"],
	["rail_hardpoints", "coolant_recycler", "lithovore"],
]
const MUTATION_SEQUENCE := [
	{"id": "reinforced_bonds", "name": "REINFORCED BONDS"},
	{"id": "efficient_metabolism", "name": "EFFICIENT METABOLISM"},
	{"id": "regenerative_matrix", "name": "REGENERATIVE MATRIX"},
	{"id": "locomotor_lattice", "name": "LOCOMOTOR LATTICE"},
	{"id": "sensory_crown", "name": "SENSORY CROWN"},
]
const NEED_RESOURCE := {
	"food": "biomass", "water": "fluid", "minerals": "mineral",
	"medicine": "spore", "knowledge": "phase",
}
const BUILD_KINDS := ["shelter", "scent_den", "root_node", "phase_anchor", "sentry"]

var rng := RandomNumberGenerator.new()
var neural_world: Node2D
var camera: Camera2D
var player: Node2D
var player_family := 0
var player_velocity := Vector2.ZERO
var player_traits: Array[String] = []
var entities: Array[Dictionary] = []
var projectiles: Array[Dictionary] = []
var particles: Array[Dictionary] = []
var recent_events: Array[String] = []
var essence := 0.0
var construction_mass := 0.0
var inventory := {"biomass": 0.0, "mineral": 0.0, "fluid": 0.0, "spore": 0.0, "phase": 0.0}
var reputation: Dictionary = {}
var discovered_societies: Dictionary = {}
var discovered_biomes: Dictionary = {}
var generation_count := 0
var objective_stage := 0
var delivered_total := 0.0
var built_count := 0
var mutation_rank := 0
var upgrade_history: Array[String] = []
var attack_cooldown := 0.0
var utility_cooldown := 0.0
var spawn_clock := 0.0
var sim_time := 0.0
var paused := false
var started := false
var capture_output_path := ""
var capture_frame_count := 0
var trace_output_path := ""
var trace_step := 0
var trace_records: Array = []
var morphology_gallery_mode := false

var ui_layer: CanvasLayer
var selection_overlay: Control
var integrity_bar: ProgressBar
var neural_bar: ProgressBar
var circulation_bar: ProgressBar
var respiration_bar: ProgressBar
var energy_bar: ProgressBar
var resource_label: Label
var biome_label: Label
var context_label: Label
var objective_label: Label
var traits_label: Label
var event_label: Label
var population_label: Label


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	rng.seed = WORLD_SEED
	var morphology_audit_path := _argument_value("--creature-stage-morphology-audit=")
	if not morphology_audit_path.is_empty():
		call_deferred("_run_morphology_audit", morphology_audit_path)
		return
	_build_world()
	_build_camera()
	_build_ui()
	_spawn_initial_ecology()
	var capture_path := _argument_value("--creature-stage-capture=")
	var morphology_capture_path := _argument_value("--creature-stage-morphology-capture=")
	var trace_path := _argument_value("--creature-stage-trace=")
	if "--creature-stage-smoke" in OS.get_cmdline_user_args():
		_start_as_family(0)
		call_deferred("_run_smoke")
	elif not morphology_capture_path.is_empty():
		_setup_morphology_gallery()
		capture_output_path = morphology_capture_path
		set_process(true)
	elif not capture_path.is_empty():
		_start_as_family(0)
		capture_output_path = capture_path
		set_process(true)
		construction_mass = 2.0
		player.aim_command = Vector2(0.85, -0.52).normalized()
		_try_build()
	elif not trace_path.is_empty():
		_start_as_family(0)
		trace_output_path = trace_path
		call_deferred("_run_trace_offline")
	elif "--creature-stage-demo" in OS.get_cmdline_user_args():
		_start_as_family(0)
	else:
		_show_selection()
	queue_redraw()


func _process(_delta: float) -> void:
	if capture_output_path.is_empty():
		return
	capture_frame_count += 1
	if capture_frame_count < 2:
		return
	var absolute_path := capture_output_path if capture_output_path.is_absolute_path() else ProjectSettings.globalize_path(capture_output_path)
	capture_output_path = ""
	DirAccess.make_dir_recursive_absolute(absolute_path.get_base_dir())
	var image := get_viewport().get_texture().get_image()
	var error := image.save_png(absolute_path)
	print("CREATURE_STAGE_CAPTURE_%s %s" % ["OK" if error == OK else "FAILED", absolute_path])
	get_tree().quit(0 if error == OK else 1)


func _argument_value(prefix: String) -> String:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with(prefix):
			return argument.trim_prefix(prefix)
	return ""


func _build_world() -> void:
	neural_world = World.new()
	neural_world.configure(WORLD_SEED)
	add_child(neural_world)


func _build_camera() -> void:
	camera = Camera2D.new()
	camera.enabled = true
	camera.position_smoothing_enabled = true
	camera.position_smoothing_speed = 8.5
	add_child(camera)


func _build_ui() -> void:
	ui_layer = CanvasLayer.new()
	ui_layer.layer = 20
	add_child(ui_layer)
	var vignette := ColorRect.new()
	vignette.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	vignette.color = Color(0.01, 0.015, 0.025, 0.06)
	vignette.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ui_layer.add_child(vignette)

	var top := Panel.new()
	top.position = Vector2(18, 17)
	top.size = Vector2(484, 117)
	top.add_theme_stylebox_override("panel", _panel_style(Color("#071019d9"), Color("#244154aa")))
	ui_layer.add_child(top)
	_label(top, Vector2(17, 8), Vector2(300, 20), "BODY SYSTEMS", Color("#a5bed0"), 10)
	integrity_bar = _meter(top, Vector2(17, 31), Vector2(212, 10), Color("#ff647d"))
	neural_bar = _meter(top, Vector2(246, 31), Vector2(212, 10), Color("#d579ff"))
	_label(top, Vector2(17, 43), Vector2(212, 17), "INTEGRITY", Color("#7d94a7"), 8)
	_label(top, Vector2(246, 43), Vector2(212, 17), "NEURAL COHERENCE", Color("#7d94a7"), 8)
	circulation_bar = _meter(top, Vector2(17, 67), Vector2(136, 8), Color("#ff496d"))
	respiration_bar = _meter(top, Vector2(169, 67), Vector2(136, 8), Color("#58e2ff"))
	energy_bar = _meter(top, Vector2(321, 67), Vector2(137, 8), Color("#c7ff56"))
	_label(top, Vector2(17, 78), Vector2(136, 17), "CIRCULATION", Color("#7d94a7"), 7)
	_label(top, Vector2(169, 78), Vector2(136, 17), "RESPIRATION", Color("#7d94a7"), 7)
	_label(top, Vector2(321, 78), Vector2(137, 17), "ENERGY", Color("#7d94a7"), 7)

	var right := Panel.new()
	right.position = Vector2(931, 17)
	right.size = Vector2(331, 146)
	right.add_theme_stylebox_override("panel", _panel_style(Color("#071019d9"), Color("#244154aa")))
	ui_layer.add_child(right)
	biome_label = _label(right, Vector2(15, 9), Vector2(301, 22), "BIOME // --", Color("#65eaff"), 11)
	population_label = _label(right, Vector2(15, 31), Vector2(301, 18), "ACTIVE ECOLOGY // --", Color("#8098aa"), 8)
	resource_label = _label(right, Vector2(15, 55), Vector2(301, 20), "ESSENCE 0  //  MATTER 0", Color("#d7e9f3"), 10)
	objective_label = _label(right, Vector2(15, 81), Vector2(301, 48), "OBJECTIVE // FEED, SURVIVE, FIND A SETTLEMENT", Color("#b5ff64"), 8)
	objective_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART

	var bottom := Panel.new()
	bottom.position = Vector2(18, 591)
	bottom.size = Vector2(545, 109)
	bottom.add_theme_stylebox_override("panel", _panel_style(Color("#071019d9"), Color("#24415488")))
	ui_layer.add_child(bottom)
	context_label = _label(bottom, Vector2(16, 9), Vector2(513, 40), "", Color("#e7f8ff"), 9)
	context_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	traits_label = _label(bottom, Vector2(16, 50), Vector2(513, 20), "", Color("#75dfff"), 8)
	event_label = _label(bottom, Vector2(16, 74), Vector2(513, 20), "WASD MOVE // LMB ATTACK // E USE // Q UTILITY // F BUILD // R MUTATE // SPACE SPRINT", Color("#70889c"), 7)

	selection_overlay = Control.new()
	selection_overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	ui_layer.add_child(selection_overlay)


func _show_selection() -> void:
	selection_overlay.visible = true
	for child in selection_overlay.get_children():
		child.queue_free()
	var shade := ColorRect.new()
	shade.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	shade.color = Color("#02050bea")
	selection_overlay.add_child(shade)
	var title := _label(selection_overlay, Vector2(178, 83), Vector2(924, 58), "CHOOSE A LINEAGE", Color("#f2fbff"), 33)
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	var sub := _label(selection_overlay, Vector2(220, 139), Vector2(840, 39), "Each body, metabolism, neural instinct, and path into society is different.", Color("#89a7b9"), 11)
	sub.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	for family_id in range(5):
		var button := Button.new()
		button.position = Vector2(91 + family_id * 224, 236)
		button.size = Vector2(202, 240)
		button.text = "%s\n\n%s" % [Neural.FAMILIES[family_id].to_upper(), _family_blurb(family_id)]
		button.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		button.add_theme_font_size_override("font_size", 12)
		button.add_theme_color_override("font_color", Neural.FAMILY_COLORS[family_id])
		button.add_theme_color_override("font_hover_color", Color.WHITE)
		button.add_theme_stylebox_override("normal", _panel_style(Color("#07111df2"), Color(Neural.FAMILY_COLORS[family_id], 0.38)))
		button.add_theme_stylebox_override("hover", _panel_style(Color(Neural.FAMILY_COLORS[family_id], 0.14), Neural.FAMILY_COLORS[family_id]))
		button.pressed.connect(_start_as_family.bind(family_id))
		selection_overlay.add_child(button)
	var footer := _label(selection_overlay, Vector2(300, 530), Vector2(680, 40), "A persistent neural field extends beyond the camera. Distant ecologies, societies, and migrations continue at cohort scale.", Color("#647d90"), 9)
	footer.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	footer.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART


func _family_blurb(family_id: int) -> String:
	return [
		"Manipulator / social\nBuild, trade, improvise",
		"Predator / pack\nTrack, pounce, consume",
		"Rooted / colonial\nPhotosynthesize, spread",
		"Phase / transmuter\nBlink, distort, feed on flux",
		"Tool chassis / lithovore\nMine, shoot, construct",
	][family_id]


func _setup_morphology_gallery() -> void:
	for record in entities:
		var body: Node2D = record["body"]
		if is_instance_valid(body):
			body.queue_free()
	entities.clear()
	morphology_gallery_mode = true
	neural_world.visible = false
	selection_overlay.visible = false
	ui_layer.visible = false
	for family_id in range(5):
		for morphotype_id in range(4):
			var seed := 0x5A170000 + family_id * 0x1000 + morphotype_id
			var blueprint := Neural.decode_morphology(family_id, seed, 0)
			var body := Creature.new()
			body.configure(blueprint)
			body.position = Vector2(-480 + family_id * 240, -225 + morphotype_id * 150)
			body.z_index = 5
			add_child(body)
			var label := Label.new()
			label.position = body.position + Vector2(-98, 55)
			label.size = Vector2(196, 20)
			label.text = "%s // %s" % [str(blueprint["family"]).to_upper(), str(blueprint["morphotype"]).to_upper()]
			label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
			label.add_theme_font_size_override("font_size", 9)
			label.add_theme_color_override("font_color", Neural.FAMILY_COLORS[family_id])
			label.z_index = 20
			add_child(label)
	camera.position = Vector2.ZERO
	started = false
	queue_redraw()


func _start_as_family(family_id: int) -> void:
	if is_instance_valid(player):
		player.queue_free()
	player_family = family_id
	player_traits.clear()
	for trait_id in FAMILY_TRAITS[family_id]:
		player_traits.append(trait_id)
	var blueprint := Neural.decode_morphology(family_id, WORLD_SEED ^ 0xA17E0000 ^ family_id, generation_count)
	player = Creature.new()
	player.configure(blueprint)
	player.position = Vector2.ZERO
	player.selected = true
	player.z_index = 10
	add_child(player)
	selection_overlay.visible = false
	started = true
	paused = false
	_log_event("You awaken as %s generation %d." % [Neural.FAMILIES[family_id], generation_count])
	_update_ui()


func _spawn_initial_ecology() -> void:
	for index in range(54):
		var family_id := index % 5
		var angle := rng.randf_range(0.0, TAU)
		var distance := rng.randf_range(180.0, 1450.0)
		_spawn_entity(family_id, Vector2.from_angle(angle) * distance, index)


func _spawn_entity(family_id: int, position_value: Vector2, serial: int, parent_policy: Dictionary = {}) -> void:
	if entities.size() >= MAX_ACTIVE_CREATURES:
		return
	var seed := WORLD_SEED ^ (serial * 0x45D9F3B) ^ (family_id * 0x1F123BB5) ^ (generation_count * 0x9E37)
	var body := Creature.new()
	body.configure(Neural.decode_morphology(family_id, seed, generation_count))
	body.position = position_value
	body.z_index = 4
	add_child(body)
	var policy := Neural.make_policy(family_id, seed) if parent_policy.is_empty() else Neural.mutate_policy(parent_policy, seed)
	entities.append({
		"body": body,
		"family_id": family_id,
		"policy": policy,
		"velocity": Vector2.ZERO,
		"age": rng.randf_range(3.0, 65.0),
		"reproduction_cooldown": rng.randf_range(45.0, 140.0),
		"attack_cooldown": rng.randf_range(0.0, 2.0),
		"target": {},
		"lineage": seed,
	})


func _physics_process(delta: float) -> void:
	if not started or paused or not is_instance_valid(player):
		return
	sim_time += delta
	attack_cooldown = maxf(0.0, attack_cooldown - delta)
	utility_cooldown = maxf(0.0, utility_cooldown - delta)
	_update_player(delta)
	_update_entities(delta)
	_update_projectiles(delta)
	_update_particles(delta)
	neural_world.ensure_chunks(player.position)
	neural_world.simulate_cohorts(delta)
	discovered_biomes[neural_world.current_biome(player.position)] = true
	_advance_objective()
	_materialize_ecology(delta)
	camera.position = player.position
	_update_ui()
	queue_redraw()


func _update_player(delta: float) -> void:
	var move := Input.get_vector("move_left", "move_right", "move_up", "move_down")
	var aim: Vector2 = player.get_global_mouse_position() - player.position
	var sprint: bool = Input.is_action_pressed("dash") and player.energy > 0.06
	var attack := 1.0 if Input.is_action_pressed("attack") else 0.0
	var feed := 1.0 if Input.is_key_pressed(KEY_E) else 0.0
	var utility := 1.0 if Input.is_key_pressed(KEY_Q) else 0.0
	_apply_player_control(delta, move, aim, attack, feed, utility, sprint)


func _apply_player_control(delta: float, move: Vector2, aim: Vector2, attack: float, feed: float, utility: float, sprint: bool) -> void:
	var speed := 128.0 * float(player.genes.get("locomotion", 1.0)) * (1.55 if sprint else 1.0)
	var locomotor_factor := clampf(player.alive_fraction() * 1.15, 0.12, 1.0)
	var target_velocity := move * speed * locomotor_factor
	player_velocity = player_velocity.lerp(target_velocity, 1.0 - exp(-delta * 10.0))
	player.position += player_velocity * delta
	player.position += neural_world.structure_collision_force(player.position, player.body_radius, player_family)
	if sprint and move.length_squared() > 0.1:
		player.energy = maxf(0.0, player.energy - delta * 0.012)
	player.set_commands(move, aim, feed, attack, utility)
	player.simulate_body(delta)
	if attack > 0.0 and attack_cooldown <= 0.0:
		_perform_attack(player, player_family, aim.normalized(), true)
	if feed > 0.5:
		_try_feed_or_interact(delta)
	if utility > 0.5 and utility_cooldown <= 0.0:
		_perform_utility()


func _update_entities(delta: float) -> void:
	var next: Array[Dictionary] = []
	for record in entities:
		var body: Node2D = record["body"]
		if not is_instance_valid(body):
			continue
		if body.dead:
			body.simulate_body(delta)
			record["age"] = float(record["age"]) + delta
			if float(record["age"]) < 180.0:
				next.append(record)
			continue
		var family_id := int(record["family_id"])
		var food_types := _accepted_resources(family_id)
		var food: Dictionary = neural_world.nearest_resource(body.position, food_types, 620.0)
		var prey: Node2D = _nearest_creature(body.position, body, _prey_families(family_id), 560.0)
		var threat: Node2D = _nearest_creature(body.position, body, _threat_families(family_id), 440.0)
		var mate: Node2D = _nearest_creature(body.position, body, [family_id], 520.0)
		var food_vec := _relative_input(body.position, food.get("pos", body.position)) if not food.is_empty() else Vector2.ZERO
		var prey_vec := _relative_input(body.position, prey.position) if is_instance_valid(prey) else Vector2.ZERO
		var threat_vec := _relative_input(body.position, threat.position) if is_instance_valid(threat) else Vector2.ZERO
		var mate_vec := _relative_input(body.position, mate.position) if is_instance_valid(mate) else Vector2.ZERO
		var player_vec := _relative_input(body.position, player.position)
		var field := Neural.world_field(body.position, WORLD_SEED)
		var inputs := PackedFloat32Array([
			food_vec.x, food_vec.y, prey_vec.x, prey_vec.y,
			threat_vec.x, threat_vec.y, mate_vec.x, mate_vec.y,
			body.energy * 2.0 - 1.0, body.alive_fraction() * 2.0 - 1.0,
			field[0] * 2.0 - 1.0, _local_crowding(body.position) * 2.0 - 1.0,
			clampf(float(record["age"]) / 120.0, 0.0, 1.0),
			sin(sim_time * 0.04), sin(float(record["lineage"]) + sim_time * 0.31),
			player_vec.x, player_vec.y, body.neural_capacity() * 2.0 - 1.0,
		])
		var outputs := Neural.policy_step(record["policy"], inputs)
		var desired := Vector2(outputs[0], outputs[1]).limit_length(1.0)
		if desired.length_squared() < 0.04:
			desired = Vector2.from_angle(sin(float(record["lineage"]) + sim_time * 0.13) * PI)
		var speed: float = float([78.0, 104.0, 22.0, 88.0, 72.0][family_id])
		if family_id == 2:
			desired *= 0.22
		var velocity: Vector2 = record["velocity"]
		velocity = velocity.lerp(desired * speed, 1.0 - exp(-delta * 3.4))
		velocity += _separation_force(body, family_id) * delta * 75.0
		body.position += velocity * delta
		body.position += neural_world.structure_collision_force(body.position, body.body_radius, family_id)
		record["velocity"] = velocity
		var aim := desired if desired.length_squared() > 0.01 else Vector2.UP
		body.set_commands(desired, aim, outputs[2], outputs[3], outputs[5])
		body.simulate_body(delta)
		record["age"] = float(record["age"]) + delta
		record["reproduction_cooldown"] = maxf(0.0, float(record["reproduction_cooldown"]) - delta)
		record["attack_cooldown"] = maxf(0.0, float(record["attack_cooldown"]) - delta)
		if outputs[2] > 0.15 and not food.is_empty() and body.position.distance_to(food["pos"]) < 34.0:
			var gained: float = neural_world.consume_resource(str(food["id"]), delta * 0.13)
			body.energy = minf(1.0, body.energy + gained * 0.45)
		if outputs[3] > 0.12 and is_instance_valid(prey) and body.position.distance_to(prey.position) < 76.0 and float(record["attack_cooldown"]) <= 0.0:
			_perform_attack(body, family_id, (prey.position - body.position).normalized(), false)
			record["attack_cooldown"] = [1.4, 0.8, 3.0, 1.6, 1.2][family_id]
		if outputs[4] > 0.25 and is_instance_valid(mate) and body.position.distance_to(mate.position) < 62.0:
			_try_reproduce(record, body)
		next.append(record)
	entities = next


func _perform_attack(source: Node2D, family_id: int, direction: Vector2, friendly: bool) -> void:
	if direction.length_squared() < 0.01:
		direction = Vector2.UP
	source.trigger_action(1.0)
	match family_id:
		0:
			_spawn_projectile(source.position + direction * 24.0, direction * 420.0, 0.25, 4.2, friendly, Color("#5feaff"), source)
			attack_cooldown = 0.34 if friendly else attack_cooldown
		1:
			_melee_arc(source, direction, 48.0, 0.28, friendly)
			attack_cooldown = 0.46 if friendly else attack_cooldown
		2:
			_spawn_projectile(source.position + direction * 22.0, direction * 230.0, 0.16, 6.5, friendly, Color("#a9ff58"), source, true)
			attack_cooldown = 0.72 if friendly else attack_cooldown
		3:
			for offset in [-0.18, 0.0, 0.18]:
				_spawn_projectile(source.position, direction.rotated(offset) * 310.0, 0.18, 5.0, friendly, Color("#b987ff"), source, true)
			attack_cooldown = 0.58 if friendly else attack_cooldown
		4:
			_spawn_projectile(source.position + direction * 30.0, direction * 650.0, 0.38, 3.5, friendly, Color("#ffb84b"), source)
			_spawn_projectile(source.position + direction * 30.0 + Vector2(-direction.y, direction.x) * 8.0, direction * 650.0, 0.32, 3.5, friendly, Color("#ff6c52"), source)
			attack_cooldown = 0.42 if friendly else attack_cooldown


func _spawn_projectile(position_value: Vector2, velocity: Vector2, damage: float, radius: float, friendly: bool, color: Color, owner: Node2D, organic := false) -> void:
	projectiles.append({
		"pos": position_value, "previous": position_value, "velocity": velocity,
		"damage": damage, "radius": radius, "life": 2.4,
		"friendly": friendly, "color": color, "owner": owner,
		"organic": organic,
	})


func _melee_arc(source: Node2D, direction: Vector2, reach: float, damage: float, friendly: bool) -> void:
	var targets: Array[Node2D] = []
	if friendly:
		for record in entities:
			targets.append(record["body"])
	else:
		targets.append(player)
	for target in targets:
		if not is_instance_valid(target) or target.dead:
			continue
		var target_delta: Vector2 = target.position - source.position
		if target_delta.length() <= reach + target.body_radius and direction.dot(target_delta.normalized()) > 0.2:
			target.damage_at(target.position - target_delta.normalized() * target.body_radius * 0.25, 8.0, damage, direction * 18.0)
			_spawn_hit_particles(target.position, Neural.FAMILY_COLORS[target.family_id], 7)


func _perform_utility() -> void:
	utility_cooldown = 4.0
	match player_family:
		0:
			construction_mass += 0.5
			_log_event("You assemble a temporary signal cairn.")
		1:
			player.energy = minf(1.0, player.energy + 0.08)
			_log_event("Pack resonance sharpens nearby prey signals.")
		2:
			player.heal(0.08)
			_log_event("Vascular sap seals damaged cells.")
		3:
			player.position += player.aim_command * 125.0
			_log_event("Nonlocal tissue folds across the ground plane.")
		4:
			construction_mass += 1.0
			player.heal(0.04)
			_log_event("Repair utility converts stored mineral into chassis cells.")


func _try_feed_or_interact(delta: float) -> void:
	var city: Dictionary = neural_world.nearest_settlement(player.position, 110.0)
	if not city.is_empty():
		var society: Dictionary = neural_world.societies[city["society_id"]]
		discovered_societies[city["society_id"]] = true
		reputation[city["society_id"]] = float(reputation.get(city["society_id"], 0.0)) + delta * 0.2
		var needed_type := str(NEED_RESOURCE.get(str(city["need"]), "biomass"))
		var carried := float(inventory.get(needed_type, 0.0))
		if carried > 0.001:
			var delivered := minf(carried, delta * 0.42)
			inventory[needed_type] = carried - delivered
			city["stores"][needed_type] = float(city["stores"].get(needed_type, 0.0)) + delivered
			delivered_total += delivered
			reputation[city["society_id"]] = float(reputation.get(city["society_id"], 0.0)) + delivered * 0.55
			player.energy = minf(1.0, player.energy + delivered * 0.08)
		context_label.text = "%s // %s tradition // needs %s // carrying %.2f // hold E to deliver" % [society["name"], society["trait"], needed_type, float(inventory.get(needed_type, 0.0))]
		return
	var accepted := _accepted_resources(player_family)
	var resource: Dictionary = neural_world.nearest_resource(player.position, accepted, 54.0)
	if resource.is_empty():
		return
	var amount: float = neural_world.consume_resource(str(resource["id"]), delta * 0.28)
	if amount <= 0.0:
		return
	player.energy = minf(1.0, player.energy + amount * 0.55)
	var resource_type := str(resource["type"])
	inventory[resource_type] = float(inventory.get(resource_type, 0.0)) + amount * 0.42
	if str(resource["type"]) in ["mineral", "biomass"]:
		construction_mass += amount * 0.7
	essence += amount * (0.45 if str(resource["type"]) in ["spore", "phase"] else 0.08)


func _try_build() -> void:
	var cost := 1.25 + float(built_count) * 0.2
	if construction_mass < cost:
		_log_event("Construction requires %.1f matter." % cost)
		return
	var direction: Vector2 = player.aim_command if player.aim_command.length_squared() > 0.01 else Vector2.UP
	var record: Dictionary = neural_world.build_structure(player.position + direction * 74.0, player_family, BUILD_KINDS[player_family], "player")
	if record.is_empty():
		_log_event("The local substrate rejects that construction site.")
		return
	construction_mass -= cost
	built_count += 1
	_log_event("Built %s from %.1f matter." % [str(record["kind"]).replace("_", " "), cost])


func _try_mutate() -> void:
	var cost := 3.0 + float(mutation_rank) * 1.25
	if essence < cost:
		_log_event("Mutation requires %.1f essence." % cost)
		return
	var upgrade: Dictionary = MUTATION_SEQUENCE[mutation_rank % MUTATION_SEQUENCE.size()]
	essence -= cost
	mutation_rank += 1
	upgrade_history.append(str(upgrade["name"]))
	player.apply_gene_upgrade(str(upgrade["id"]))
	_log_event("Neural germline accepted: %s." % str(upgrade["name"]))


func _advance_objective() -> void:
	var previous := objective_stage
	match objective_stage:
		0:
			var carried_total := 0.0
			for type in inventory:
				carried_total += float(inventory[type])
			if carried_total >= 0.4:
				objective_stage = 1
		1:
			if not discovered_societies.is_empty():
				objective_stage = 2
		2:
			if delivered_total >= 0.5:
				objective_stage = 3
		3:
			if built_count > 0:
				objective_stage = 4
		4:
			if mutation_rank > 0:
				objective_stage = 5
		5:
			if discovered_biomes.size() >= 3:
				objective_stage = 6
	if objective_stage != previous:
		_log_event("Expedition stage %d complete." % (previous + 1))


func _objective_text() -> String:
	return [
		"HARVEST // assimilate and carry 0.4 compatible matter",
		"CONTACT // locate and enter a generated settlement",
		"RECIPROCITY // deliver 0.5 of the settlement's requested resource",
		"HABITAT // gather matter and press F to construct",
		"GERMLINE // gather essence and press R to mutate",
		"MIGRATION // discover three neural-field biomes",
		"OPEN WORLD // form alliances, hunt, build, mutate, and migrate",
	][mini(objective_stage, 6)]


func _trace_control(step: int) -> Dictionary:
	var movement_cycle := [Vector2.UP, Vector2(0.75, -0.65), Vector2.RIGHT, Vector2(0.7, 0.7), Vector2.DOWN, Vector2(-0.75, 0.65), Vector2.LEFT, Vector2(-0.7, -0.7)]
	var move: Vector2 = movement_cycle[floori(float(step) / 30.0) % movement_cycle.size()]
	var aim := Vector2.from_angle(-PI * 0.5 + float(step) * 0.047)
	return {
		"move": move,
		"aim": aim,
		"attack": 1.0 if step % 19 < 2 else 0.0,
		"feed": 1.0 if step % 31 < 6 else 0.0,
		"utility": 1.0 if step in [42, 126, 210] else 0.0,
		"sprint": step % 60 >= 45,
	}


func _trace_control_json(control: Dictionary) -> Dictionary:
	var move: Vector2 = control["move"]
	var aim: Vector2 = control["aim"]
	return {
		"move": [move.x, move.y],
		"aim": [aim.x, aim.y],
		"attack": control["attack"],
		"feed": control["feed"],
		"utility": control["utility"],
		"sprint": control["sprint"],
	}


func _trace_state() -> Dictionary:
	var status: Dictionary = player.status_snapshot()
	var field: PackedFloat32Array = Neural.world_field(player.position, WORLD_SEED)
	var field_values: Array[float] = []
	for value in field:
		field_values.append(value)
	var chunk: Vector2i = neural_world.world_to_chunk(player.position)
	var nearest: Dictionary = neural_world.nearest_resource(player.position, [], 260.0)
	var resource_context := {"type": "none", "relative": [0.0, 0.0], "amount": 0.0}
	if not nearest.is_empty():
		var relative: Vector2 = Vector2(nearest["pos"]) - player.position
		resource_context = {
			"type": str(nearest["type"]),
			"relative": [relative.x, relative.y],
			"amount": float(nearest["amount"]),
		}
	return {
		"position": [player.position.x, player.position.y],
		"velocity": [player_velocity.x, player_velocity.y],
		"status": status,
		"genes": player.genes.duplicate(true),
		"organ_alive": player.organ_alive.duplicate(true),
		"organ_totals": player.organ_totals.duplicate(true),
		"world_field": field_values,
		"chunk": [chunk.x, chunk.y],
		"biome": neural_world.current_biome(player.position),
		"nearest_resource": resource_context,
		"inventory": inventory.duplicate(true),
		"essence": essence,
		"construction_mass": construction_mass,
		"objective_stage": objective_stage,
		"active_creatures": entities.size() + 1,
		"active_projectiles": projectiles.size(),
		"built_structures": neural_world.structures.size(),
		"known_societies": discovered_societies.size(),
	}


func _run_trace_offline() -> void:
	# This is the deterministic teacher rollout. It deliberately bypasses wall
	# clock scheduling while executing the same public player/world transitions.
	for step in range(TRACE_STEPS):
		trace_step = step
		var before := _trace_state()
		var control := _trace_control(step)
		sim_time += TRACE_DELTA
		attack_cooldown = maxf(0.0, attack_cooldown - TRACE_DELTA)
		utility_cooldown = maxf(0.0, utility_cooldown - TRACE_DELTA)
		_apply_player_control(TRACE_DELTA, control["move"], control["aim"], float(control["attack"]), float(control["feed"]), float(control["utility"]), bool(control["sprint"]))
		_update_entities(TRACE_DELTA)
		_update_projectiles(TRACE_DELTA)
		_update_particles(TRACE_DELTA)
		neural_world.ensure_chunks(player.position)
		neural_world.simulate_cohorts(TRACE_DELTA)
		discovered_biomes[neural_world.current_biome(player.position)] = true
		_advance_objective()
		_materialize_ecology(TRACE_DELTA)
		trace_records.append({
			"step": step,
			"dt": TRACE_DELTA,
			"before": before,
			"action": _trace_control_json(control),
			"after": _trace_state(),
		})
	_finish_trace()


func _finish_trace() -> void:
	var transition_json := JSON.stringify(trace_records, "", false)
	var hashing := HashingContext.new()
	hashing.start(HashingContext.HASH_SHA256)
	hashing.update(transition_json.to_utf8_buffer())
	var transition_sha256 := hashing.finish().hex_encode()
	var report := {
		"format": "nullvector-creature-stage-causal-trace-v1",
		"world_seed": WORLD_SEED,
		"family": Neural.FAMILIES[player_family],
		"fixed_hz": 30,
		"transition_count": trace_records.size(),
		"transition_sha256": transition_sha256,
		"contracts": {
			"morphology": "coordinate-conditioned-safe-scaffold-v1",
			"controller": "recurrent-18x12x10-v1",
			"world_field": "continuous-5-channel-latent-v1",
			"physiology": "cellular-organ-causal-scaffold-v1",
			"action": "move2-aim2-attack-feed-utility-sprint-v1",
		},
		"transitions": trace_records,
	}
	var absolute_path := trace_output_path if trace_output_path.is_absolute_path() else ProjectSettings.globalize_path(trace_output_path)
	trace_output_path = ""
	DirAccess.make_dir_recursive_absolute(absolute_path.get_base_dir())
	var handle := FileAccess.open(absolute_path, FileAccess.WRITE)
	if handle == null:
		push_error("Unable to write creature-stage trace: " + absolute_path)
		get_tree().quit(1)
		return
	handle.store_string(JSON.stringify(report, "  ", false))
	handle.close()
	print("CREATURE_STAGE_TRACE_OK steps=%d sha256=%s" % [trace_records.size(), transition_sha256])
	get_tree().quit(0)


func _try_reproduce(record: Dictionary, body: Node2D) -> void:
	if float(record["reproduction_cooldown"]) > 0.0 or body.energy < 0.88 or float(record["age"]) < 50.0:
		return
	if entities.size() >= MAX_ACTIVE_CREATURES:
		return
	body.energy -= 0.32
	record["reproduction_cooldown"] = 95.0 / float(body.genes.get("fertility", 1.0))
	generation_count += 1
	var offset := Vector2.from_angle(rng.randf_range(0.0, TAU)) * 42.0
	_spawn_entity(int(record["family_id"]), body.position + offset, rng.randi(), record["policy"])
	_spawn_hit_particles(body.position, Neural.FAMILY_COLORS[int(record["family_id"])], 12)


func _update_projectiles(delta: float) -> void:
	var next: Array[Dictionary] = []
	for shot in projectiles:
		shot["life"] = float(shot["life"]) - delta
		if float(shot["life"]) <= 0.0:
			continue
		shot["previous"] = shot["pos"]
		shot["pos"] = Vector2(shot["pos"]) + Vector2(shot["velocity"]) * delta
		var hit := false
		var targets: Array[Node2D] = []
		if bool(shot["friendly"]):
			for record in entities:
				targets.append(record["body"])
		else:
			targets.append(player)
		for target in targets:
			if not is_instance_valid(target) or target == shot["owner"] or target.dead:
				continue
			if target.position.distance_to(shot["pos"]) <= target.body_radius + float(shot["radius"]):
				var impacts: int = target.damage_at(shot["pos"], float(shot["radius"]) + 4.0, float(shot["damage"]), Vector2(shot["velocity"]).normalized() * 18.0)
				if impacts > 0:
					_spawn_hit_particles(shot["pos"], shot["color"], 8)
					hit = true
					if bool(shot["friendly"]) and target.dead:
						essence += 0.35
						construction_mass += target.cells.size() * 0.006
					break
		if not hit:
			next.append(shot)
	projectiles = next


func _spawn_hit_particles(position_value: Vector2, color: Color, count: int) -> void:
	for _index in range(count):
		particles.append({
			"pos": position_value,
			"velocity": Vector2.from_angle(rng.randf_range(0.0, TAU)) * rng.randf_range(18.0, 90.0),
			"life": rng.randf_range(0.25, 0.75), "max_life": 0.75,
			"color": color, "size": rng.randf_range(1.0, 3.2),
		})


func _update_particles(delta: float) -> void:
	var next: Array[Dictionary] = []
	for particle in particles:
		particle["life"] = float(particle["life"]) - delta
		if float(particle["life"]) <= 0.0:
			continue
		particle["velocity"] = Vector2(particle["velocity"]) * exp(-delta * 4.0)
		particle["pos"] = Vector2(particle["pos"]) + Vector2(particle["velocity"]) * delta
		next.append(particle)
	particles = next


func _materialize_ecology(delta: float) -> void:
	spawn_clock -= delta
	if spawn_clock > 0.0 or entities.size() >= 62:
		return
	spawn_clock = 2.5
	var chunk: Vector2i = neural_world.world_to_chunk(player.position)
	var candidate_coord: Vector2i = chunk + Vector2i(rng.randi_range(-2, 2), rng.randi_range(-2, 2))
	var data: Dictionary = neural_world.chunks.get(candidate_coord, {})
	var cohorts: Array = data.get("cohorts", [])
	if cohorts.is_empty():
		return
	var cohort: Dictionary = cohorts[rng.randi_range(0, cohorts.size() - 1)]
	var position_value: Vector2 = Vector2(candidate_coord) * World.CHUNK_SIZE + Vector2(rng.randf_range(30.0, World.CHUNK_SIZE - 30.0), rng.randf_range(30.0, World.CHUNK_SIZE - 30.0))
	_spawn_entity(int(cohort["family_id"]), position_value, rng.randi())


func _nearest_creature(position_value: Vector2, self_body: Node2D, families: Array, max_distance: float) -> Node2D:
	var best: Node2D
	var best_distance := max_distance
	if is_instance_valid(player) and player != self_body and player.family_id in families and not player.dead:
		var distance := position_value.distance_to(player.position)
		if distance < best_distance:
			best = player
			best_distance = distance
	for record in entities:
		var candidate: Node2D = record["body"]
		if not is_instance_valid(candidate) or candidate == self_body or candidate.dead or candidate.family_id not in families:
			continue
		var distance := position_value.distance_to(candidate.position)
		if distance < best_distance:
			best = candidate
			best_distance = distance
	return best


func _prey_families(family_id: int) -> Array:
	return [[1, 2, 4], [0, 1, 2], [], [0, 1, 4], [0, 1, 3]][family_id]


func _threat_families(family_id: int) -> Array:
	return [[1, 3, 4], [0, 1, 3, 4], [0, 1, 4], [4], [0, 3]][family_id]


func _accepted_resources(family_id: int) -> Array:
	return [
		["biomass", "fluid", "spore", "mineral"],
		["biomass", "fluid"],
		["fluid", "spore"],
		["phase", "spore"],
		["mineral", "fluid", "phase"],
	][family_id]


func _relative_input(from: Vector2, to: Vector2) -> Vector2:
	return ((to - from) / 260.0).limit_length(1.0)


func _local_crowding(position_value: Vector2) -> float:
	var count := 0
	for record in entities:
		var body: Node2D = record["body"]
		if is_instance_valid(body) and not body.dead and body.position.distance_to(position_value) < 130.0:
			count += 1
	return clampf(float(count) / 8.0, 0.0, 1.0)


func _separation_force(body: Node2D, family_id: int) -> Vector2:
	var force := Vector2.ZERO
	for record in entities:
		var other: Node2D = record["body"]
		if not is_instance_valid(other) or other == body or other.dead:
			continue
		var separation_delta: Vector2 = body.position - other.position
		var min_distance: float = (body.body_radius + other.body_radius) * (0.55 if other.family_id == family_id else 0.82)
		var distance: float = separation_delta.length()
		if distance > 0.1 and distance < min_distance:
			force += separation_delta.normalized() * (1.0 - distance / min_distance)
	return force.limit_length(1.0)


func _update_ui() -> void:
	if not is_instance_valid(player):
		return
	var state: Dictionary = player.status_snapshot()
	integrity_bar.value = float(state["integrity"]) * 100.0
	neural_bar.value = float(state["neural"]) * 100.0
	circulation_bar.value = float(state["circulation"]) * 100.0
	respiration_bar.value = float(state["respiration"]) * 100.0
	energy_bar.value = float(state["energy"]) * 100.0
	biome_label.text = "BIOME // " + neural_world.current_biome(player.position).to_upper()
	population_label.text = "ACTIVE ECOLOGY // %d BODIES  //  EPOCH %d" % [entities.size() + 1, neural_world.simulation_epoch]
	resource_label.text = "ESS %.1f // MAT %.1f // BIO %.1f MIN %.1f FLU %.1f" % [essence, construction_mass, float(inventory["biomass"]), float(inventory["mineral"]), float(inventory["fluid"])]
	var trait_names: Array[String] = []
	for trait_id in player_traits:
		trait_names.append(str(TRAITS[trait_id]["name"]).to_upper())
	for upgrade in upgrade_history:
		trait_names.append(upgrade)
	traits_label.text = "TRAITS // " + "  ·  ".join(trait_names)
	var nearest: Dictionary = neural_world.nearest_resource(player.position, _accepted_resources(player_family), 180.0)
	var city: Dictionary = neural_world.nearest_settlement(player.position, 360.0)
	if not city.is_empty():
		var society: Dictionary = neural_world.societies[city["society_id"]]
		context_label.text = "%s // %s // %s tradition // need: %s" % [society["name"], society["ethos"], society["trait"], city["need"]]
	elif not nearest.is_empty():
		context_label.text = "SENSED // %s at %.0fm // hold E nearby to assimilate" % [str(nearest["type"]).to_upper(), player.position.distance_to(nearest["pos"]) / 10.0]
	else:
		context_label.text = "SENSORY FIELD QUIET // migrate toward luminous substrate signals"
	objective_label.text = "OBJECTIVE %d/6 // %s" % [mini(objective_stage + 1, 6), _objective_text()]


func _log_event(text: String) -> void:
	recent_events.push_front(text)
	if recent_events.size() > 5:
		recent_events.resize(5)
	if is_instance_valid(event_label):
		event_label.text = text


func _draw() -> void:
	if morphology_gallery_mode:
		draw_rect(Rect2(Vector2(-640, -360), VIEW_SIZE), Color("#061419"), true)
		for x in range(-600, 601, 40):
			draw_line(Vector2(x, -360), Vector2(x, 360), Color(0.12, 0.32, 0.34, 0.08), 1.0)
		for y in range(-320, 321, 40):
			draw_line(Vector2(-640, y), Vector2(640, y), Color(0.12, 0.32, 0.34, 0.08), 1.0)
	for shot in projectiles:
		var color: Color = shot["color"]
		draw_line(shot["previous"], shot["pos"], Color(color, 0.35), float(shot["radius"]) * 1.8)
		draw_circle(shot["pos"], float(shot["radius"]), color)
	for particle in particles:
		var alpha := clampf(float(particle["life"]) / float(particle["max_life"]), 0.0, 1.0)
		draw_circle(particle["pos"], float(particle["size"]), Color(particle["color"], alpha))
	if is_instance_valid(player):
		var aim: Vector2 = player.aim_command
		draw_line(player.position + aim * 35.0, player.position + aim * 62.0, Color(Neural.FAMILY_COLORS[player_family], 0.48), 1.0)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_ESCAPE:
			paused = not paused
			get_tree().paused = paused
		if event.keycode >= KEY_1 and event.keycode <= KEY_5 and not started:
			_start_as_family(int(event.keycode - KEY_1))
		if started and not paused and event.keycode == KEY_F:
			_try_build()
		if started and not paused and event.keycode == KEY_R:
			_try_mutate()


func _panel_style(background: Color, border: Color) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = background
	style.border_color = border
	style.set_border_width_all(1)
	style.corner_radius_top_left = 5
	style.corner_radius_top_right = 5
	style.corner_radius_bottom_left = 5
	style.corner_radius_bottom_right = 5
	return style


func _label(parent: Node, position_value: Vector2, size_value: Vector2, text_value: String, color: Color, font_size: int) -> Label:
	var label := Label.new()
	label.position = position_value
	label.size = size_value
	label.text = text_value
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	parent.add_child(label)
	return label


func _meter(parent: Node, position_value: Vector2, size_value: Vector2, color: Color) -> ProgressBar:
	var meter := ProgressBar.new()
	meter.position = position_value
	meter.size = size_value
	meter.min_value = 0.0
	meter.max_value = 100.0
	meter.value = 100.0
	meter.show_percentage = false
	meter.add_theme_stylebox_override("background", _panel_style(Color("#101b25"), Color.TRANSPARENT))
	meter.add_theme_stylebox_override("fill", _panel_style(color, Color.TRANSPARENT))
	parent.add_child(meter)
	return meter


func _run_morphology_audit(output_path: String) -> void:
	var required_organs := [
		["brain", "heart", "lung", "gut", "eye"],
		["brain", "heart", "lung", "gut", "eye"],
		["meristem", "vascular", "frond", "bulb", "photoreceptor"],
		["phase_brain", "flux", "orbital", "transmuter", "singularity"],
		["processor", "coolant_pump", "radiator", "battery", "optic"],
	]
	var specimens: Array = []
	var failures: Array[String] = []
	var family_summary: Dictionary = {}
	for family_id in range(5):
		var signatures: Dictionary = {}
		var morphotypes: Dictionary = {}
		var symmetry_sum := 0.0
		var minimum_symmetry := 1.0
		var maximum_cells := 0
		var minimum_cells := 99999
		for morphotype_id in range(4):
			for sample in range(8):
				var seed := 0x5A170000 + family_id * 0x1000 + sample * 4 + morphotype_id
				var blueprint := Neural.decode_morphology(family_id, seed, 0)
				var metrics: Dictionary = Neural.analyze_morphology(blueprint)
				metrics["seed"] = seed
				specimens.append(metrics)
				signatures[str(metrics["signature"])] = true
				morphotypes[str(metrics["morphotype"])] = true
				symmetry_sum += float(metrics["symmetry"])
				minimum_symmetry = minf(minimum_symmetry, float(metrics["symmetry"]))
				maximum_cells = maxi(maximum_cells, int(metrics["cell_count"]))
				minimum_cells = mini(minimum_cells, int(metrics["cell_count"]))
				if int(metrics["morphotype_id"]) != morphotype_id:
					failures.append("family%d seed%d morphotype mismatch" % [family_id, seed])
				if float(metrics["connected_fraction"]) < 0.99999:
					failures.append("family%d seed%d disconnected %.4f" % [family_id, seed, float(metrics["connected_fraction"])])
				if not bool(metrics["vertical_ordered"]):
					failures.append("family%d seed%d violates 2.5D organ order" % [family_id, seed])
				if int(metrics["cell_count"]) < 70 or int(metrics["cell_count"]) > 500:
					failures.append("family%d seed%d cell count %d" % [family_id, seed, int(metrics["cell_count"])])
				if float(metrics["symmetry"]) < 0.48:
					failures.append("family%d seed%d symmetry %.4f" % [family_id, seed, float(metrics["symmetry"])])
				var organs: Array = metrics["organs"]
				for required in required_organs[family_id]:
					if required not in organs:
						failures.append("family%d seed%d missing %s" % [family_id, seed, required])
		family_summary[str(family_id)] = {
			"family": Neural.FAMILIES[family_id],
			"specimen_count": 32,
			"morphotype_count": morphotypes.size(),
			"unique_signature_count": signatures.size(),
			"mean_symmetry": symmetry_sum / 32.0,
			"minimum_symmetry": minimum_symmetry,
			"cell_count_range": [minimum_cells, maximum_cells],
		}
		if morphotypes.size() != 4:
			failures.append("family%d lacks all four morphotypes" % family_id)
		if signatures.size() < 8:
			failures.append("family%d signature diversity %d < 8" % [family_id, signatures.size()])
	var report := {
		"format": "nullvector-creature-stage-morphology-audit-v1",
		"passed": failures.is_empty(),
		"specimen_count": specimens.size(),
		"family_count": 5,
		"morphotypes_per_family": 4,
		"family_summary": family_summary,
		"failures": failures,
		"specimens": specimens,
	}
	var absolute_path := output_path if output_path.is_absolute_path() else ProjectSettings.globalize_path(output_path)
	DirAccess.make_dir_recursive_absolute(absolute_path.get_base_dir())
	var handle := FileAccess.open(absolute_path, FileAccess.WRITE)
	if handle == null:
		push_error("Unable to write morphology audit: " + absolute_path)
		get_tree().quit(1)
		return
	handle.store_string(JSON.stringify(report, "  ", false))
	handle.close()
	print("CREATURE_STAGE_MORPHOLOGY_%s specimens=%d failures=%d" % ["OK" if failures.is_empty() else "FAILED", specimens.size(), failures.size()])
	get_tree().quit(0 if failures.is_empty() else 1)


func _run_smoke() -> void:
	var spawn_failures: Array[String] = []
	for record in entities:
		var initial_body: Node2D = record["body"]
		if initial_body.neural_capacity() <= 0.0 or initial_body.circulation_capacity() <= 0.0 or initial_body.digestion_capacity() <= 0.0:
			spawn_failures.append("%s:n%.2f:c%.2f:d%.2f" % [str(record["lineage"]), initial_body.neural_capacity(), initial_body.circulation_capacity(), initial_body.digestion_capacity()])
	for _frame in range(120):
		_update_entities(1.0 / 30.0)
		neural_world.simulate_cohorts(1.0 / 30.0)
	var family_counts := [0, 0, 0, 0, 0]
	var dead_count := 0
	var family_signatures: Dictionary = {}
	for record in entities:
		var body: Node2D = record["body"]
		var record_family := int(record["family_id"])
		family_counts[record_family] += 1
		if not family_signatures.has(record_family):
			family_signatures[record_family] = _morphology_signature(body)
		if body.dead:
			dead_count += 1
	var unique_shapes: Dictionary = {}
	for family_id in family_signatures:
		unique_shapes[str(family_signatures[family_id])] = true
	construction_mass = 2.0
	essence = 3.0
	player.aim_command = Vector2.UP
	_try_build()
	_try_mutate()
	var structure_pos: Vector2 = neural_world.structures[0]["pos"] if not neural_world.structures.is_empty() else Vector2.ZERO
	var hostile_collision: float = neural_world.structure_collision_force(structure_pos + Vector2.RIGHT, 20.0, 1).length()
	var friendly_collision: float = neural_world.structure_collision_force(structure_pos + Vector2.RIGHT, 20.0, 0).length()
	var passed: bool = entities.size() >= 50 and spawn_failures.is_empty() and dead_count <= 4 and unique_shapes.size() == 5 and neural_world.societies.size() >= 1 and neural_world.structures.size() == 1 and mutation_rank == 1 and hostile_collision > 1.0 and friendly_collision < 0.001
	print("CREATURE_STAGE_SMOKE_%s active=%d families=%s chunks=%d societies=%d spawn_failures=%s deaths=%d" % ["OK" if passed else "FAILED", entities.size(), str(family_counts), neural_world.chunks.size(), neural_world.societies.size(), str(spawn_failures), dead_count])
	var report := {
		"format": "nullvector-creature-stage-scaffold-smoke-v1",
		"passed": passed,
		"world_seed": WORLD_SEED,
		"active_creatures": entities.size() + 1,
		"family_counts": family_counts,
		"family_signatures": family_signatures,
		"unique_shape_count": unique_shapes.size(),
		"active_chunks": neural_world.chunks.size(),
		"society_count": neural_world.societies.size(),
		"structure_count": neural_world.structures.size(),
		"mutation_rank": mutation_rank,
		"upgrade_history": upgrade_history,
		"hostile_structure_collision": hostile_collision,
		"friendly_structure_collision": friendly_collision,
		"spawn_failures": spawn_failures,
		"deaths_after_4_seconds": dead_count,
		"player": player.status_snapshot(),
		"neural_contract": {
			"morphology": "coordinate-conditioned-safe-scaffold-v1",
			"controller": "recurrent-18x12x10-v1",
			"world_field": "continuous-5-channel-latent-v1",
		},
	}
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--creature-stage-report="):
			var output_path := argument.trim_prefix("--creature-stage-report=")
			var absolute_path := output_path if output_path.is_absolute_path() else ProjectSettings.globalize_path(output_path)
			DirAccess.make_dir_recursive_absolute(absolute_path.get_base_dir())
			var handle := FileAccess.open(absolute_path, FileAccess.WRITE)
			if handle != null:
				handle.store_string(JSON.stringify(report, "  ", false))
				handle.close()
	get_tree().quit(0 if passed else 1)


func _morphology_signature(body: Node2D) -> String:
	var min_grid := Vector2i(999, 999)
	var max_grid := Vector2i(-999, -999)
	var appendages: Dictionary = {}
	var organs: Dictionary = {}
	for cell in body.cells:
		var grid: Vector2i = cell["grid"]
		min_grid.x = mini(min_grid.x, grid.x)
		min_grid.y = mini(min_grid.y, grid.y)
		max_grid.x = maxi(max_grid.x, grid.x)
		max_grid.y = maxi(max_grid.y, grid.y)
		var appendage := int(cell.get("appendage", -1))
		if appendage >= 0:
			appendages[appendage] = true
		var organ := str(cell.get("organ", "none"))
		if organ != "none":
			organs[organ] = true
	return "%dx%d:c%d:a%d:o%d" % [max_grid.x - min_grid.x + 1, max_grid.y - min_grid.y + 1, body.cells.size(), appendages.size(), organs.size()]
