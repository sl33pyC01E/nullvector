extends Node2D

const Neural = preload("res://scripts/creature_stage/creature_neural.gd")
const Creature = preload("res://scripts/creature_stage/neural_creature.gd")
const World = preload("res://scripts/creature_stage/neural_world.gd")

const WORLD_SEED := 0x4E554C4C
const MAX_ACTIVE_CREATURES := 86
const VIEW_SIZE := Vector2(1280, 720)
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
var reputation: Dictionary = {}
var discovered_societies: Dictionary = {}
var generation_count := 0
var attack_cooldown := 0.0
var utility_cooldown := 0.0
var spawn_clock := 0.0
var sim_time := 0.0
var paused := false
var started := false

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
	_build_world()
	_build_camera()
	_build_ui()
	_spawn_initial_ecology()
	if "--creature-stage-smoke" in OS.get_cmdline_user_args():
		_start_as_family(0)
		call_deferred("_run_smoke")
	elif "--creature-stage-demo" in OS.get_cmdline_user_args():
		_start_as_family(0)
	else:
		_show_selection()
	queue_redraw()


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
	event_label = _label(bottom, Vector2(16, 74), Vector2(513, 20), "WASD MOVE  //  LMB ATTACK  //  E FEED/INTERACT  //  Q UTILITY  //  SPACE SPRINT", Color("#70889c"), 7)

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
	_materialize_ecology(delta)
	camera.position = player.position
	_update_ui()
	queue_redraw()


func _update_player(delta: float) -> void:
	var move := Input.get_vector("move_left", "move_right", "move_up", "move_down")
	var aim: Vector2 = player.get_global_mouse_position() - player.position
	var sprint: bool = Input.is_action_pressed("dash") and player.energy > 0.06
	var speed := 128.0 * (1.55 if sprint else 1.0)
	var locomotor_factor := clampf(player.alive_fraction() * 1.15, 0.12, 1.0)
	var target_velocity := move * speed * locomotor_factor
	player_velocity = player_velocity.lerp(target_velocity, 1.0 - exp(-delta * 10.0))
	player.position += player_velocity * delta
	if sprint and move.length_squared() > 0.1:
		player.energy = maxf(0.0, player.energy - delta * 0.012)
	var attack := 1.0 if Input.is_action_pressed("attack") else 0.0
	player.set_commands(move, aim, 0.0, attack, 0.0)
	player.simulate_body(delta)
	if attack > 0.0 and attack_cooldown <= 0.0:
		_perform_attack(player, player_family, aim.normalized(), true)
	if Input.is_key_pressed(KEY_E):
		_try_feed_or_interact(delta)
	if Input.is_key_pressed(KEY_Q) and utility_cooldown <= 0.0:
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
		if construction_mass >= 1.0:
			construction_mass -= delta * 0.6
			player.energy = minf(1.0, player.energy + delta * 0.18)
			city["stores"]["mineral"] = float(city["stores"]["mineral"]) + delta * 0.6
		context_label.text = "%s // %s tradition // needs %s // population %d // hold E to trade matter" % [society["name"], society["trait"], city["need"], city["population"]]
		return
	var accepted := _accepted_resources(player_family)
	var resource: Dictionary = neural_world.nearest_resource(player.position, accepted, 54.0)
	if resource.is_empty():
		return
	var amount: float = neural_world.consume_resource(str(resource["id"]), delta * 0.28)
	if amount <= 0.0:
		return
	player.energy = minf(1.0, player.energy + amount * 0.55)
	if str(resource["type"]) in ["mineral", "biomass"]:
		construction_mass += amount * 0.7
	essence += amount * (0.45 if str(resource["type"]) in ["spore", "phase"] else 0.08)


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
	resource_label.text = "ESSENCE %.1f  //  MATTER %.1f  //  GEN %d" % [essence, construction_mass, generation_count]
	var trait_names: Array[String] = []
	for trait_id in player_traits:
		trait_names.append(str(TRAITS[trait_id]["name"]).to_upper())
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
	if discovered_societies.size() > 0:
		objective_label.text = "OBJECTIVE // GROW ESSENCE TO 3.0 AND PRODUCE A MUTATED DESCENDANT"
	if essence >= 3.0:
		objective_label.text = "MUTATION READY // NEXT DESCENDANT INHERITS A POLICY AND BODY VARIANT"


func _log_event(text: String) -> void:
	recent_events.push_front(text)
	if recent_events.size() > 5:
		recent_events.resize(5)
	if is_instance_valid(event_label):
		event_label.text = text


func _draw() -> void:
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
	var passed: bool = entities.size() >= 50 and spawn_failures.is_empty() and dead_count <= 4 and unique_shapes.size() == 5 and neural_world.societies.size() >= 1
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
