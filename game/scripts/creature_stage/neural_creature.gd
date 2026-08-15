class_name NeuralCreature
extends Node2D

const Neural = preload("res://scripts/creature_stage/creature_neural.gd")

signal died(creature: NeuralCreature)
signal cell_lost(creature: NeuralCreature, tissue: String, organ: String)

var family_id := 0
var family := "humanoid"
var genome_seed := 0
var generation := 0
var genes: Dictionary = {}
var cells: Array = []
var organ_totals: Dictionary = {}
var organ_alive: Dictionary = {}
var fluids: Array = []

var move_command := Vector2.ZERO
var aim_command := Vector2.UP
var feed_command := 0.0
var attack_command := 0.0
var utility_command := 0.0
var locomotion_phase := 0.0
var action_impulse := 0.0
var energy := 0.72
var hydration := 0.85
var age := 0.0
var dead := false
var selected := false
var neural_glitch := 0.0
var body_radius := 30.0
var display_scale := 1.0
var base_color := Color.WHITE


func configure(blueprint: Dictionary) -> void:
	family_id = int(blueprint.get("family_id", 0))
	family = str(blueprint.get("family", Neural.FAMILIES[family_id]))
	genome_seed = int(blueprint.get("seed", 0))
	generation = int(blueprint.get("generation", 0))
	genes = blueprint.get("genes", {}).duplicate(true)
	cells = blueprint.get("cells", []).duplicate(true)
	base_color = Neural.FAMILY_COLORS[family_id]
	_rebuild_organs()
	_recalculate_radius()
	queue_redraw()


func _rebuild_organs() -> void:
	organ_totals.clear()
	organ_alive.clear()
	for cell in cells:
		var organ := str(cell.get("organ", "none"))
		if organ != "none":
			organ_totals[organ] = int(organ_totals.get(organ, 0)) + 1
			if bool(cell.get("alive", true)):
				organ_alive[organ] = int(organ_alive.get(organ, 0)) + 1


func _recalculate_radius() -> void:
	body_radius = 12.0
	for cell in cells:
		var rest: Vector2 = cell["rest"]
		body_radius = maxf(body_radius, rest.length() + 5.0)


func set_commands(move: Vector2, aim: Vector2, feed: float, attack: float, utility: float) -> void:
	move_command = move.limit_length(1.0)
	if aim.length_squared() > 0.01:
		aim_command = aim.normalized()
	feed_command = clampf(feed, 0.0, 1.0)
	attack_command = clampf(attack, 0.0, 1.0)
	utility_command = clampf(utility, 0.0, 1.0)


func trigger_action(strength := 1.0) -> void:
	action_impulse = maxf(action_impulse, strength)


func simulate_body(delta: float) -> void:
	age += delta
	var locomotor := move_command.length()
	locomotion_phase = fmod(locomotion_phase + delta * lerpf(2.2, 8.0, locomotor), TAU)
	action_impulse = move_toward(action_impulse, 0.0, delta * 3.2)
	neural_glitch = move_toward(neural_glitch, 1.0 - neural_capacity(), delta * 0.8)
	if dead:
		_simulate_corpse(delta)
	else:
		_simulate_living_cells(delta)
		_simulate_physiology(delta)
	_simulate_fluids(delta)
	queue_redraw()


func _simulate_living_cells(delta: float) -> void:
	var speed_phase := sin(locomotion_phase)
	for cell in cells:
		if not bool(cell.get("alive", true)):
			continue
		var rest: Vector2 = cell["rest"]
		var target := rest
		var appendage := int(cell.get("appendage", -1))
		var side := int(cell.get("side", 0))
		var tissue := str(cell.get("tissue", "skin"))
		if appendage >= 0:
			var phase_offset := float(appendage % 2) * PI
			var stride := sin(locomotion_phase + phase_offset)
			var leverage := clampf(abs(rest.y) / 30.0 + abs(rest.x) / 36.0, 0.2, 1.0)
			if tissue in ["locomotor", "root"]:
				target.x += stride * 4.2 * leverage * move_command.length()
				target.y += cos(locomotion_phase + phase_offset) * 1.7 * leverage * move_command.length()
			elif tissue in ["weapon", "skin", "phase"]:
				target.x += stride * 1.8 * leverage * move_command.length()
		if str(cell.get("organ", "none")) in ["eye", "optic", "photoreceptor", "singularity"]:
			target += aim_command * 1.2
		if attack_command > 0.2 and tissue in ["weapon", "locomotor", "phase"]:
			target += aim_command * action_impulse * 5.0
		if family_id == 2 and tissue in ["skin", "root"]:
			target.x += sin(age * 1.8 + rest.y * 0.13) * 0.55
		if family_id == 3 and str(cell.get("organ", "")) == "orbital":
			target += Vector2(cos(age * 1.7 + appendage), sin(age * 1.3 + appendage)) * 1.4
		var health := float(cell.get("health", 1.0))
		var stiffness := lerpf(4.0, 18.0, health) * float(genes.get("bond_strength", 1.0))
		var pos: Vector2 = cell["pos"]
		var velocity: Vector2 = cell["vel"]
		velocity += (target - pos) * stiffness * delta
		velocity *= exp(-delta * lerpf(4.0, 9.0, health))
		if neural_glitch > 0.25:
			velocity += Vector2(
				sin(age * 19.0 + rest.x), cos(age * 15.0 + rest.y)
			) * neural_glitch * 4.0 * delta
		pos += velocity
		cell["vel"] = velocity
		cell["pos"] = pos


func _simulate_physiology(delta: float) -> void:
	var metabolic_cost := (0.0016 + move_command.length() * 0.0032 + attack_command * 0.0045) * float(genes.get("metabolism", 1.0))
	var circulation := circulation_capacity()
	var respiration := respiration_capacity()
	energy = maxf(0.0, energy - metabolic_cost * delta)
	hydration = maxf(0.0, hydration - 0.0007 * delta)
	if family_id == 2:
		energy = minf(1.0, energy + 0.0022 * delta * maxf(0.15, respiration))
	if family_id == 3:
		energy = minf(1.0, energy + 0.0014 * delta)
	var systemic_stress := maxf(0.0, 0.30 - circulation) + maxf(0.0, 0.28 - respiration)
	if energy <= 0.0:
		systemic_stress += 0.2
	if systemic_stress > 0.0:
		for cell in cells:
			if bool(cell.get("alive", true)):
				cell["health"] = maxf(0.0, float(cell.get("health", 1.0)) - systemic_stress * delta * 0.025)
				if float(cell["health"]) <= 0.0:
					_kill_cell(cell)
	if neural_capacity() < 0.08 or alive_fraction() < 0.18:
		_die()


func _simulate_corpse(delta: float) -> void:
	for cell in cells:
		var velocity: Vector2 = cell.get("vel", Vector2.ZERO)
		velocity *= exp(-delta * 4.5)
		velocity.y += delta * 0.8
		cell["pos"] = Vector2(cell["pos"]) + velocity
		cell["vel"] = velocity


func _simulate_fluids(delta: float) -> void:
	var next: Array = []
	for drop in fluids:
		var life := float(drop.get("life", 0.0)) - delta
		if life <= 0.0:
			continue
		var velocity: Vector2 = drop.get("velocity", Vector2.ZERO)
		velocity *= exp(-delta * 3.4)
		var pos: Vector2 = drop.get("pos", Vector2.ZERO) + velocity * delta
		drop["pos"] = pos
		drop["velocity"] = velocity
		drop["life"] = life
		drop["radius"] = minf(float(drop.get("radius", 1.0)) + delta * 1.9, 5.5)
		next.append(drop)
	fluids = next


func damage_at(world_point: Vector2, radius: float, amount: float, impulse := Vector2.ZERO) -> int:
	if dead:
		return 0
	var local_point := to_local(world_point)
	var hits := 0
	for cell in cells:
		if not bool(cell.get("alive", true)):
			continue
		var pos: Vector2 = cell["pos"]
		if pos.distance_to(local_point) <= radius:
			var health := float(cell.get("health", 1.0)) - amount
			cell["health"] = health
			cell["vel"] = Vector2(cell.get("vel", Vector2.ZERO)) + impulse * 0.025
			hits += 1
			_spawn_fluid(pos, str(cell.get("tissue", "skin")), impulse)
			if health <= 0.0:
				_kill_cell(cell)
	if hits > 0:
		_rebuild_organs()
		if neural_capacity() < 0.08 or alive_fraction() < 0.18:
			_die()
		queue_redraw()
	return hits


func cut_segment(world_a: Vector2, world_b: Vector2, width := 2.2) -> int:
	var a := to_local(world_a)
	var b := to_local(world_b)
	var hits := 0
	for cell in cells:
		if not bool(cell.get("alive", true)):
			continue
		var p: Vector2 = cell["pos"]
		var ab := b - a
		var t := clampf((p - a).dot(ab) / maxf(ab.length_squared(), 0.001), 0.0, 1.0)
		if p.distance_to(a + ab * t) <= width:
			cell["health"] = 0.0
			_kill_cell(cell)
			_spawn_fluid(p, str(cell.get("tissue", "skin")), Vector2(-ab.y, ab.x).normalized() * 8.0)
			hits += 1
	if hits > 0:
		_rebuild_organs()
		if neural_capacity() < 0.08 or alive_fraction() < 0.18:
			_die()
	return hits


func heal(amount: float) -> void:
	if dead:
		return
	var repair := amount * float(genes.get("repair", 1.0))
	for cell in cells:
		if bool(cell.get("alive", true)):
			cell["health"] = minf(1.0, float(cell.get("health", 1.0)) + repair)


func apply_gene_upgrade(upgrade_id: String) -> void:
	match upgrade_id:
		"reinforced_bonds":
			genes["bond_strength"] = float(genes.get("bond_strength", 1.0)) * 1.18
			heal(0.14)
		"efficient_metabolism":
			genes["metabolism"] = maxf(0.45, float(genes.get("metabolism", 1.0)) * 0.86)
			energy = minf(1.0, energy + 0.18)
		"regenerative_matrix":
			genes["repair"] = float(genes.get("repair", 1.0)) * 1.28
			heal(0.22)
		"locomotor_lattice":
			genes["locomotion"] = float(genes.get("locomotion", 1.0)) * 1.16
		"sensory_crown":
			genes["sense_range"] = float(genes.get("sense_range", 1.0)) * 1.22
	queue_redraw()


func _kill_cell(cell: Dictionary) -> void:
	if not bool(cell.get("alive", true)):
		return
	cell["alive"] = false
	cell["health"] = 0.0
	cell["vel"] = Vector2(cell.get("vel", Vector2.ZERO)) + Vector2(randf_range(-1.2, 1.2), randf_range(-0.6, 1.4))
	cell_lost.emit(self, str(cell.get("tissue", "skin")), str(cell.get("organ", "none")))


func _spawn_fluid(local_pos: Vector2, tissue: String, impulse: Vector2) -> void:
	var color := Color("#ff4166")
	if family_id == 2:
		color = Color("#83ed62")
	elif family_id == 3:
		color = Color("#aa6dff")
	elif family_id == 4:
		color = Color("#37b6d0") if tissue != "weapon" else Color("#ff9a3c")
	fluids.append({
		"pos": local_pos,
		"velocity": impulse * 0.16 + Vector2(randf_range(-3.0, 3.0), randf_range(-3.0, 3.0)),
		"life": randf_range(2.5, 6.0),
		"max_life": 6.0,
		"radius": 0.8,
		"color": color,
	})


func _die() -> void:
	if dead:
		return
	dead = true
	move_command = Vector2.ZERO
	for cell in cells:
		cell["vel"] = Vector2(cell.get("vel", Vector2.ZERO)) + Vector2(randf_range(-0.7, 0.7), randf_range(-0.25, 0.65))
	died.emit(self)


func organ_capacity(names: Array[String]) -> float:
	var total := 0
	var alive := 0
	for name in names:
		total += int(organ_totals.get(name, 0))
		alive += int(organ_alive.get(name, 0))
	return float(alive) / float(maxi(total, 1))


func neural_capacity() -> float:
	return organ_capacity(["brain", "meristem", "phase_brain", "processor"])


func circulation_capacity() -> float:
	return organ_capacity(["heart", "vascular", "flux", "coolant_pump"])


func respiration_capacity() -> float:
	return organ_capacity(["lung", "frond", "orbital", "radiator"])


func digestion_capacity() -> float:
	return organ_capacity(["gut", "bulb", "transmuter", "battery"])


func sensory_capacity() -> float:
	return organ_capacity(["eye", "photoreceptor", "singularity", "optic"])


func alive_fraction() -> float:
	var alive := 0
	for cell in cells:
		if bool(cell.get("alive", true)):
			alive += 1
	return float(alive) / float(maxi(cells.size(), 1))


func status_snapshot() -> Dictionary:
	return {
		"family": family,
		"generation": generation,
		"integrity": alive_fraction(),
		"neural": neural_capacity(),
		"circulation": circulation_capacity(),
		"respiration": respiration_capacity(),
		"digestion": digestion_capacity(),
		"senses": sensory_capacity(),
		"energy": energy,
		"hydration": hydration,
		"dead": dead,
	}


func _draw() -> void:
	# The body never rotates. This shadow defines the 2.5D ground plane.
	draw_ellipse(Vector2(0, body_radius * 0.38), Vector2(body_radius * 0.82, body_radius * 0.26), Color(0.0, 0.0, 0.0, 0.42))
	for drop in fluids:
		var color: Color = drop["color"]
		var alpha := clampf(float(drop["life"]) / float(drop["max_life"]), 0.0, 1.0)
		draw_circle(drop["pos"] + Vector2(0, body_radius * 0.26), float(drop["radius"]), Color(color, alpha * 0.26))

	var ordered := cells.duplicate()
	ordered.sort_custom(func(a: Dictionary, b: Dictionary) -> bool: return Vector2(a["pos"]).y < Vector2(b["pos"]).y)
	var pulse := 0.5 + sin(age * 3.2) * 0.5
	for cell in ordered:
		var pos: Vector2 = cell["pos"]
		var alive := bool(cell.get("alive", true))
		var health := float(cell.get("health", 0.0))
		var tissue := str(cell.get("tissue", "skin"))
		var color := Neural.tissue_color(tissue, family_id, health, pulse)
		if not alive:
			color = color.darkened(0.62)
		var size := 1.65 if tissue not in ["sensor", "neural"] else 1.9
		draw_circle(pos + Vector2(0.8, 1.2), size + 0.35, Color(0.0, 0.0, 0.0, 0.5))
		draw_circle(pos, size, color)
		if health < 0.55 and alive:
			draw_arc(pos, size + 0.35, 0.0, TAU * health, 8, Color("#ff9b72"), 0.45)

	if selected:
		draw_arc(Vector2.ZERO, body_radius + 5.0, 0.0, TAU, 48, Color(base_color, 0.78), 1.0)
		var sense := 34.0 + sensory_capacity() * 56.0
		var angle := aim_command.angle()
		draw_colored_polygon(PackedVector2Array([
			Vector2.ZERO,
			aim_command.rotated(-0.42) * sense,
			aim_command.rotated(0.42) * sense,
		]), Color(base_color, 0.045))
		draw_arc(Vector2.ZERO, sense, angle - 0.42, angle + 0.42, 20, Color(base_color, 0.22), 1.0)


func draw_ellipse(center: Vector2, radius: Vector2, color: Color) -> void:
	var points := PackedVector2Array()
	for index in range(32):
		var angle := TAU * float(index) / 32.0
		points.append(center + Vector2(cos(angle) * radius.x, sin(angle) * radius.y))
	draw_colored_polygon(points, color)
