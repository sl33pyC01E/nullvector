extends Node2D

const ATLAS_PATH := "res://generated/anatomical_demo/v1/neural_motion_atlas.png"
const ANATOMY_PATH := "res://generated/anatomical_demo/v1/anatomy.json"
const MANIFEST_PATH := "res://generated/anatomical_demo/v1/manifest.json"
const FAMILIES := ["HUMANOID", "ANIMALIAN", "PLANTLIKE", "ANOMALY", "MACHINE"]
const FAMILY_COLORS := [
	Color("#4ce7ff"), Color("#ff6fb5"), Color("#9dff4f"),
	Color("#b789ff"), Color("#ffb13b")
]
const TISSUE_COLORS := {
	"skin": Color("#66d5e4"), "bone": Color("#e8e0c4"),
	"muscle": Color("#ef5572"), "vascular": Color("#ff4d78"),
	"respiratory": Color("#6ce8ff"), "digestive": Color("#ffc45a"),
	"neural": Color("#d978ff"), "sensor": Color("#efffff"),
	"storage": Color("#f2d36d"), "phase": Color("#b279ff"),
	"root": Color("#91dc66"), "machine": Color("#a8b6c4"),
	"armor": Color("#b6c3d2"), "weapon": Color("#ff755b")
}
const ORGAN_GROUPS := {
	"neural": ["brain", "phase_brain", "processor", "meristem"],
	"circulation": ["heart", "vascular", "coolant_pump", "flux"],
	"respiration": ["lung", "photoreceptor", "radiator"],
	"digestion": ["gut", "transmuter", "battery", "bulb"],
	"senses": ["eye", "photoreceptor", "singularity", "optic"],
}
const CELL_SCALE := 4.0
const SPRITE_SIZE := 192.0
const ARENA := Rect2(24, 80, 930, 570)

var atlas: Texture2D
var anatomy: Dictionary
var manifest: Dictionary
var creatures: Array[Dictionary] = []
var puddles: Array[Dictionary] = []
var fragments: Array[Dictionary] = []
var selected := 0
var paused := false
var time := 0.0
var show_neural := true
var show_cells := false
var show_organs := true
var show_skeleton := true
var show_contacts := true
var tool := "inspect"
var cut_start := Vector2.ZERO
var cut_current := Vector2.ZERO
var cutting := false
var message := "SELECT A CREATURE OR PRESS 1-5"
var message_time := 4.0


func _ready() -> void:
	atlas = load(ATLAS_PATH)
	anatomy = _read_json(ANATOMY_PATH)
	manifest = _read_json(MANIFEST_PATH)
	if atlas == null or anatomy.is_empty() or manifest.get("status", "") != "ready":
		push_error("Anatomical demo bundle is unavailable")
		return
	var positions := [Vector2(145, 230), Vector2(360, 245), Vector2(575, 240), Vector2(785, 230), Vector2(470, 475)]
	for family in range(5):
		var specimen: Dictionary = anatomy["specimens"][family]
		var health := PackedFloat32Array()
		health.resize(specimen["cells"].size())
		health.fill(1.0)
		creatures.append({
			"family": family, "specimen": specimen, "pos": positions[family],
			"vel": Vector2.ZERO, "phase": float(family) * 0.13,
			"health": health, "energy": 1.0, "hunger": 0.12,
			"action": "idle", "action_time": 0.0, "dead": false,
			"wander": float(family) * 1.47, "scar": PackedByteArray(),
		})
		var scars := PackedByteArray()
		scars.resize(health.size())
		creatures[family]["scar"] = scars
	set_process(true)
	queue_redraw()
	if "--anatomical-demo-smoke" in OS.get_cmdline_user_args():
		call_deferred("_run_smoke")


func _run_smoke() -> void:
	var errors: Array[String] = []
	if creatures.size() != 5:
		errors.append("creature family count")
	if atlas == null or atlas.get_width() != 480 or atlas.get_height() != 1536:
		errors.append("neural atlas geometry")
	for family in range(creatures.size()):
		var specimen: Dictionary = creatures[family]["specimen"]
		if specimen["cells"].size() < 100:
			errors.append("family %d cell census" % family)
		if specimen["components"].size() < 3:
			errors.append("family %d organ census" % family)
		if specimen["skeleton"]["edges"].is_empty() or specimen["skeleton"]["muscles"].is_empty():
			errors.append("family %d physical graph" % family)
	var systems := _systems(creatures[0])
	for key in ["integrity", "neural", "circulation", "respiration", "digestion", "senses"]:
		if not systems.has(key) or not is_equal_approx(float(systems[key]), 1.0):
			errors.append("healthy system %s" % key)
	if errors.is_empty():
		print("ANATOMICAL_DEMO_SMOKE_OK families=5 phases=16 atlas=480x1536 cells=true organs=true joints=true")
		get_tree().quit(0)
	else:
		for error in errors:
			push_error(error)
		get_tree().quit(1)


func _read_json(path: String) -> Dictionary:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return {}
	var value = JSON.parse_string(file.get_as_text())
	return value if value is Dictionary else {}


func _process(delta: float) -> void:
	if paused:
		queue_redraw()
		return
	time += delta
	message_time = maxf(0.0, message_time - delta)
	_update_puddles(delta)
	_update_fragments(delta)
	for index in range(creatures.size()):
		_update_creature(index, delta)
	queue_redraw()


func _update_creature(index: int, delta: float) -> void:
	var creature := creatures[index]
	var systems := _systems(creature)
	var family := int(creature["family"])
	var controlled := index == selected
	var desired := Vector2.ZERO
	if controlled:
		desired = Input.get_vector("move_left", "move_right", "move_up", "move_down")
	else:
		var wander_angle := time * (0.23 + family * 0.027) + float(creature["wander"])
		desired = Vector2(cos(wander_angle), sin(wander_angle) * 0.62)
		if fmod(time + family, 7.0) < 1.2:
			desired *= 0.15
	if bool(creature["dead"]):
		desired = Vector2.ZERO

	var locomotion := _locomotion_integrity(creature)
	var cognition := float(systems["neural"])
	if cognition < 0.55 and desired.length() > 0:
		desired = desired.rotated(sin(time * 5.7 + family) * (0.55 - cognition) * 1.2)
	var base_speed: float = [82.0, 98.0, 34.0, 72.0, 76.0][family]
	var acceleration: float = [7.5, 8.8, 3.2, 4.5, 10.0][family]
	var speed: float = base_speed * locomotion * (0.45 + cognition * 0.55)
	if family == 3:
		# Anomalies float and retain some lateral inertia.
		creature["vel"] = Vector2(creature["vel"]).lerp(desired * speed, delta * acceleration * .45)
	elif family == 2:
		# Plants pull their root plate and drag the crown behind it.
		creature["vel"] = Vector2(creature["vel"]).lerp(desired * speed, delta * acceleration)
	else:
		# Grounded bodies accelerate through planted appendages. Releasing input
		# creates rapid static friction instead of endless top-down sliding.
		creature["vel"] = Vector2(creature["vel"]).lerp(desired * speed, delta * acceleration)
		if desired.length_squared() < .01:
			creature["vel"] = Vector2(creature["vel"]).move_toward(Vector2.ZERO, delta * 260.0)
	var position: Vector2 = creature["pos"]
	position += Vector2(creature["vel"]) * delta
	position.x = clampf(position.x, ARENA.position.x + 65, ARENA.end.x - 65)
	position.y = clampf(position.y, ARENA.position.y + 85, ARENA.end.y - 65)
	creature["pos"] = position

	var moving := Vector2(creature["vel"]).length() > 4.0
	var phase_rate := (1.6 + Vector2(creature["vel"]).length() / 52.0) * locomotion
	if family == 3:
		phase_rate = 1.15
	elif family == 2:
		phase_rate *= .62
	if moving or family == 3:
		creature["phase"] = fmod(float(creature["phase"]) + delta * phase_rate, 1.0)
	else:
		creature["phase"] = fmod(float(creature["phase"]) + delta * .23, 1.0)
	creature["energy"] = clampf(float(creature["energy"]) - delta * (.002 + Vector2(creature["vel"]).length() * .000015), 0.0, 1.0)
	creature["hunger"] = clampf(float(creature["hunger"]) + delta * .0017, 0.0, 1.0)
	creature["action_time"] = maxf(0.0, float(creature["action_time"]) - delta)
	if float(systems["neural"]) < .08 or float(systems["integrity"]) < .14:
		creature["dead"] = true
		creature["action"] = "incapacitated"
	creatures[index] = creature


func _systems(creature: Dictionary) -> Dictionary:
	var specimen: Dictionary = creature["specimen"]
	var health: PackedFloat32Array = creature["health"]
	var integrity := 0.0
	var values := {"neural": [], "circulation": [], "respiration": [], "digestion": [], "senses": []}
	for index in range(health.size()):
		integrity += health[index]
		var organ := str(specimen["cells"][index].get("organ", "none"))
		for group in ORGAN_GROUPS:
			if organ in ORGAN_GROUPS[group]:
				values[group].append(health[index])
	var result := {"integrity": integrity / maxf(1.0, health.size())}
	for group in values:
		var total := 0.0
		for value in values[group]:
			total += float(value)
		result[group] = total / maxf(1.0, values[group].size()) if not values[group].is_empty() else 1.0
	return result


func _locomotion_integrity(creature: Dictionary) -> float:
	var specimen: Dictionary = creature["specimen"]
	var health: PackedFloat32Array = creature["health"]
	var total := 0.0
	var count := 0
	for index in range(health.size()):
		if int(specimen["cells"][index].get("appendage", -1)) >= 0:
			total += health[index]
			count += 1
	return total / maxf(1.0, count)


func _update_puddles(delta: float) -> void:
	for index in range(puddles.size() - 1, -1, -1):
		puddles[index]["life"] = float(puddles[index]["life"]) - delta
		puddles[index]["radius"] = minf(31.0, float(puddles[index]["radius"]) + delta * 5.5)
		if float(puddles[index]["life"]) <= 0:
			puddles.remove_at(index)


func _update_fragments(delta: float) -> void:
	for index in range(fragments.size() - 1, -1, -1):
		var fragment := fragments[index]
		fragment["life"] = float(fragment["life"]) - delta
		fragment["vel"] = Vector2(fragment["vel"]) * pow(.16, delta)
		fragment["pos"] = Vector2(fragment["pos"]) + Vector2(fragment["vel"]) * delta
		if float(fragment["life"]) <= 0:
			fragments.remove_at(index)
		else:
			fragments[index] = fragment


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		var key: int = event.physical_keycode
		if key >= KEY_1 and key <= KEY_5:
			selected = int(key - KEY_1)
			message = "%s SELECTED" % FAMILIES[selected]
			message_time = 2.0
		elif key == KEY_TAB:
			selected = (selected + 1) % creatures.size()
		elif key == KEY_SPACE:
			paused = not paused
		elif key == KEY_N:
			show_neural = not show_neural
		elif key == KEY_C:
			show_cells = not show_cells
		elif key == KEY_O:
			show_organs = not show_organs
		elif key == KEY_K:
			show_skeleton = not show_skeleton
		elif key == KEY_G:
			show_contacts = not show_contacts
		elif key == KEY_X:
			tool = "cut"
		elif key == KEY_D:
			tool = "damage"
		elif key == KEY_H:
			tool = "heal"
		elif key == KEY_I:
			tool = "inspect"
		elif key == KEY_R:
			_reset_creature(selected)
		elif key in [KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT]:
			_trigger_action(selected, key)
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			if event.pressed:
				if tool == "cut":
					cutting = true
					cut_start = event.position
					cut_current = event.position
				elif tool == "damage":
					_apply_radial(event.position, 24.0, -.42)
				elif tool == "heal":
					_apply_radial(event.position, 30.0, .48)
				else:
					_select_nearest(event.position)
			elif cutting:
				cut_current = event.position
				_apply_cut(cut_start, cut_current)
				cutting = false
	if event is InputEventMouseMotion and cutting:
		cut_current = event.position


func _trigger_action(index: int, key: int) -> void:
	var actions := {KEY_UP: "reach", KEY_DOWN: "brace", KEY_LEFT: "left actuator", KEY_RIGHT: "right actuator"}
	creatures[index]["action"] = actions[key]
	creatures[index]["action_time"] = .7
	creatures[index]["energy"] = maxf(0.0, float(creatures[index]["energy"]) - .018)
	message = "%s // %s" % [FAMILIES[index], str(actions[key]).to_upper()]
	message_time = 1.5


func _select_nearest(point: Vector2) -> void:
	var best := selected
	var distance := 9999.0
	for index in range(creatures.size()):
		var candidate := point.distance_to(creatures[index]["pos"])
		if candidate < distance:
			best = index
			distance = candidate
	if distance < 120.0:
		selected = best


func _cell_world(creature: Dictionary, index: int) -> Vector2:
	var xy: Array = creature["specimen"]["cells"][index]["xy"]
	return Vector2(creature["pos"]) + Vector2(float(xy[0]), float(xy[1])) * CELL_SCALE


func _distance_to_segment(point: Vector2, a: Vector2, b: Vector2) -> float:
	var delta := b - a
	var denominator := maxf(delta.length_squared(), .0001)
	var t := clampf((point - a).dot(delta) / denominator, 0.0, 1.0)
	return point.distance_to(a + delta * t)


func _apply_cut(from: Vector2, to: Vector2) -> void:
	var affected := 0
	for creature_index in range(creatures.size()):
		var creature := creatures[creature_index]
		var health: PackedFloat32Array = creature["health"]
		for index in range(health.size()):
			if health[index] > 0 and _distance_to_segment(_cell_world(creature, index), from, to) <= 5.5:
				health[index] = 0.0
				_detach_cell(creature, index, to - from)
				affected += 1
		creature["health"] = health
		creatures[creature_index] = creature
	message = "CUT SEVERED %d CELLS" % affected
	message_time = 2.2


func _apply_radial(center: Vector2, radius: float, amount: float) -> void:
	var affected := 0
	for creature_index in range(creatures.size()):
		var creature := creatures[creature_index]
		var health: PackedFloat32Array = creature["health"]
		var scars: PackedByteArray = creature["scar"]
		for index in range(health.size()):
			var distance := _cell_world(creature, index).distance_to(center)
			if distance <= radius:
				var strength := 1.0 - distance / radius
				var previous := health[index]
				health[index] = clampf(health[index] + amount * strength, 0.0, 1.0)
				if amount > 0 and health[index] > previous:
					scars[index] = 1
				elif previous > 0 and health[index] <= 0:
					_detach_cell(creature, index, Vector2.from_angle(time + index) * 20.0)
				affected += 1
		creature["health"] = health
		creature["scar"] = scars
		creatures[creature_index] = creature
	message = "%s // %d CELLS" % [("HEAL" if amount > 0 else "IMPACT"), affected]
	message_time = 1.7


func _detach_cell(creature: Dictionary, index: int, impulse: Vector2) -> void:
	var cell: Dictionary = creature["specimen"]["cells"][index]
	var tissue := str(cell.get("tissue", "skin"))
	var color: Color = TISSUE_COLORS.get(tissue, Color.WHITE)
	var world := _cell_world(creature, index)
	fragments.append({"pos": world, "vel": impulse.normalized() * randf_range(16.0, 42.0) + Vector2(randf_range(-12, 12), randf_range(-8, 8)), "life": 6.0, "color": color})
	if tissue in ["vascular", "respiratory", "digestive", "neural"]:
		puddles.append({"pos": world + Vector2(0, 18), "radius": 3.0, "life": 11.0, "color": color})


func _reset_creature(index: int) -> void:
	var health: PackedFloat32Array = creatures[index]["health"]
	health.fill(1.0)
	var scars: PackedByteArray = creatures[index]["scar"]
	scars.fill(0)
	creatures[index]["health"] = health
	creatures[index]["scar"] = scars
	creatures[index]["dead"] = false
	creatures[index]["energy"] = 1.0
	creatures[index]["action"] = "regrown"
	creatures[index]["action_time"] = 1.0
	message = "%s RECONSTITUTED" % FAMILIES[index]
	message_time = 2.0


func _draw() -> void:
	draw_rect(Rect2(0, 0, 1280, 720), Color("#03080d"), true)
	_draw_background()
	for puddle in puddles:
		var color: Color = puddle["color"]
		var alpha := clampf(float(puddle["life"]) / 11.0, 0.0, 1.0)
		draw_circle(puddle["pos"], float(puddle["radius"]), Color(color.r, color.g, color.b, alpha * .22))
	for index in range(creatures.size()):
		_draw_creature(index)
	for fragment in fragments:
		var color: Color = fragment["color"]
		draw_circle(fragment["pos"], 2.7, Color(color.r, color.g, color.b, clampf(float(fragment["life"]) / 2.0, 0.0, 1.0)))
	if cutting:
		draw_line(cut_start, cut_current, Color("#ff4e72"), 2.0)
		draw_circle(cut_current, 5.0, Color("#fff0f3"), false, 1.0)
	_draw_panel()


func _draw_background() -> void:
	draw_rect(ARENA, Color("#07161a"), true)
	for x in range(int(ARENA.position.x), int(ARENA.end.x), 24):
		draw_line(Vector2(x, ARENA.position.y), Vector2(x, ARENA.end.y), Color(0.12, .45, .48, .08), 1.0)
	for y in range(int(ARENA.position.y), int(ARENA.end.y), 24):
		draw_line(Vector2(ARENA.position.x, y), Vector2(ARENA.end.x, y), Color(0.12, .45, .48, .08), 1.0)
	draw_rect(ARENA, Color("#1d6170"), false, 1.0)
	draw_string(ThemeDB.fallback_font, Vector2(28, 34), "NULLVECTOR // ANATOMICAL CREATURE-STAGE VERTICAL SLICE", HORIZONTAL_ALIGNMENT_LEFT, -1, 18, Color("#e8f4f5"))
	draw_string(ThemeDB.fallback_font, Vector2(28, 59), "LIVE VAE MOTION + CELL AUTHORITY + PHYSIOLOGY + GROUNDED CONTROL", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color("#4ce7ff"))


func _draw_creature(index: int) -> void:
	var creature := creatures[index]
	var family := int(creature["family"])
	var color: Color = FAMILY_COLORS[family]
	var pos: Vector2 = creature["pos"]
	var phase := int(floor(float(creature["phase"]) * 16.0)) % 16
	var systems := _systems(creature)
	var integrity := float(systems["integrity"])
	var shadow_width := 54.0 + family * 2.0
	draw_ellipse(pos + Vector2(0, 66), Vector2(shadow_width, 13), Color(0, 0, 0, .46))
	if show_neural:
		var source := Rect2(family * 96, phase * 96, 96, 96)
		var destination := Rect2(pos - Vector2.ONE * SPRITE_SIZE * .5, Vector2.ONE * SPRITE_SIZE)
		var tint := Color(1, 1, 1, .28 + integrity * .72)
		draw_texture_rect_region(atlas, destination, source, tint)
	if show_skeleton:
		_draw_skeleton(creature, color)
	if show_organs:
		_draw_organs(creature)
	if show_cells:
		_draw_cells(creature)
	if show_contacts:
		_draw_contacts(creature, color)
	if index == selected:
		draw_arc(pos, 81.0, 0, TAU, 48, Color(color.r, color.g, color.b, .72), 1.5)
		draw_line(pos + Vector2(-30, 78), pos + Vector2(30, 78), Color(color.r, color.g, color.b, .28), 4.0)
		draw_line(pos + Vector2(-30, 78), pos + Vector2(-30 + 60 * integrity, 78), color, 4.0)
	var action := str(creature["action"]) if float(creature["action_time"]) > 0 else ("DEAD" if creature["dead"] else "living")
	draw_string(ThemeDB.fallback_font, pos + Vector2(-55, -104), "%s // %s" % [FAMILIES[family], action.to_upper()], HORIZONTAL_ALIGNMENT_CENTER, 110, 10, color)


func _draw_skeleton(creature: Dictionary, color: Color) -> void:
	var skeleton: Dictionary = creature["specimen"]["skeleton"]
	var nodes: Array = skeleton["nodes"]
	for edge in skeleton["edges"]:
		var left: Array = nodes[int(edge[0])]
		var right: Array = nodes[int(edge[1])]
		var a := Vector2(creature["pos"]) + Vector2(float(left[0]), float(left[1])) * CELL_SCALE
		var b := Vector2(creature["pos"]) + Vector2(float(right[0]), float(right[1])) * CELL_SCALE
		draw_line(a, b, Color(color.r, color.g, color.b, .34), 1.2)
	for node in nodes:
		var point := Vector2(creature["pos"]) + Vector2(float(node[0]), float(node[1])) * CELL_SCALE
		draw_circle(point, 2.0, Color("#f2f7e4"))


func _draw_organs(creature: Dictionary) -> void:
	var health: PackedFloat32Array = creature["health"]
	var specimen: Dictionary = creature["specimen"]
	for component in specimen["components"]:
		var organ := str(component["organ"])
		if organ == "none":
			continue
		var total := 0.0
		var count := 0
		for index in range(health.size()):
			if int(specimen["cells"][index]["component"]) == int(component["index"]):
				total += health[index]
				count += 1
		var value := total / maxf(1.0, count)
		var anchor: Array = component["anchor"]
		var radius: Array = component["radius"]
		var center := Vector2(creature["pos"]) + Vector2(float(anchor[0]), float(anchor[1])) * CELL_SCALE
		var color := _organ_color(organ)
		draw_arc(center, maxf(float(radius[0]), float(radius[1])) * CELL_SCALE, 0, TAU * value, 20, Color(color.r, color.g, color.b, .86), 1.6)
		draw_circle(center, 2.5, Color(color.r, color.g, color.b, .78))


func _draw_cells(creature: Dictionary) -> void:
	var health: PackedFloat32Array = creature["health"]
	var scars: PackedByteArray = creature["scar"]
	for index in range(health.size()):
		if health[index] <= 0:
			continue
		var tissue := str(creature["specimen"]["cells"][index]["tissue"])
		var color: Color = TISSUE_COLORS.get(tissue, Color.WHITE)
		color = color.darkened((1.0 - health[index]) * .72)
		if scars[index] > 0:
			color = color.lerp(Color("#d4a478"), .38)
		draw_circle(_cell_world(creature, index), 2.25, Color(color.r, color.g, color.b, .78))


func _draw_contacts(creature: Dictionary, color: Color) -> void:
	var family := int(creature["family"])
	if family == 3:
		draw_arc(Vector2(creature["pos"]) + Vector2(0, 64), 24.0 + sin(time * 2.0) * 4.0, 0, TAU, 24, Color(color.r, color.g, color.b, .32), 1.0)
		return
	var health: PackedFloat32Array = creature["health"]
	var specimen: Dictionary = creature["specimen"]
	var terminals: Dictionary = {}
	for index in range(health.size()):
		var appendage := int(specimen["cells"][index]["appendage"])
		if appendage < 0 or health[index] <= 0:
			continue
		var point := _cell_world(creature, index)
		if not terminals.has(appendage) or point.y > Vector2(terminals[appendage]).y:
			terminals[appendage] = point
	for appendage in terminals:
		var point: Vector2 = terminals[appendage]
		var planted := (int(floor(float(creature["phase"]) * 8.0)) + int(appendage)) % 2 == 0
		if family == 2:
			planted = true
		if planted:
			draw_line(point, point + Vector2(0, 8), Color(color.r, color.g, color.b, .56), 1.2)
			draw_line(point + Vector2(-5, 8), point + Vector2(5, 8), Color(color.r, color.g, color.b, .56), 1.2)


func _organ_color(organ: String) -> Color:
	for group in ORGAN_GROUPS:
		if organ in ORGAN_GROUPS[group]:
			return {"neural": Color("#d978ff"), "circulation": Color("#ff4d78"), "respiration": Color("#6ce8ff"), "digestion": Color("#ffc45a"), "senses": Color("#f4ffff")}[group]
	return Color("#9aaabd")


func _draw_panel() -> void:
	draw_rect(Rect2(972, 80, 292, 570), Color("#071017"), true)
	draw_rect(Rect2(972, 80, 292, 570), Color("#244855"), false, 1.0)
	var creature := creatures[selected] if not creatures.is_empty() else {}
	if creature.is_empty():
		return
	var family := int(creature["family"])
	var color: Color = FAMILY_COLORS[family]
	var systems := _systems(creature)
	draw_string(ThemeDB.fallback_font, Vector2(992, 111), "SELECTED // %s" % FAMILIES[family], HORIZONTAL_ALIGNMENT_LEFT, -1, 16, color)
	draw_string(ThemeDB.fallback_font, Vector2(992, 132), str(creature["specimen"]["genome_id"]), HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color("#8da6b2"))
	var y := 164.0
	for key in ["integrity", "neural", "circulation", "respiration", "digestion", "senses"]:
		_draw_meter(Vector2(992, y), key.to_upper(), float(systems[key]), color if key == "integrity" else _system_color(key))
		y += 35
	_draw_meter(Vector2(992, y), "ENERGY", float(creature["energy"]), Color("#9dff4f"))
	y += 42
	draw_string(ThemeDB.fallback_font, Vector2(992, y), "ANATOMICAL GRAPH", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color("#e5f2f4"))
	y += 21
	var specimen: Dictionary = creature["specimen"]
	var skeleton: Dictionary = specimen["skeleton"]
	var facts := [
		"%d living cells" % specimen["cells"].size(),
		"%d organ components" % specimen["components"].size(),
		"%d skeleton nodes / %d edges" % [skeleton["nodes"].size(), skeleton["edges"].size()],
		"%d antagonistic muscles" % skeleton["muscles"].size(),
		"EMA step %d // 115M parameters" % int(manifest["checkpoint"]["global_step"]),
	]
	for fact in facts:
		draw_string(ThemeDB.fallback_font, Vector2(992, y), fact, HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color("#8da6b2"))
		y += 17
	y += 7
	draw_string(ThemeDB.fallback_font, Vector2(992, y), "TOOLS", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color("#e5f2f4"))
	y += 22
	for item in [["I", "inspect"], ["X", "cut"], ["D", "damage"], ["H", "heal"]]:
		var active: bool = tool == item[1]
		var tool_color := Color("#4ce7ff") if active else Color("#718995")
		draw_string(ThemeDB.fallback_font, Vector2(992, y), "%s  %s" % [item[0], str(item[1]).to_upper()], HORIZONTAL_ALIGNMENT_LEFT, 100, 10, tool_color)
		y += 17
	var footer := "WASD MOVE  ARROWS ACTUATE  1-5 SELECT  TAB CYCLE\nN NEURAL  C CELLS  O ORGANS  K SKELETON  G CONTACTS\nR REGROW  SPACE PAUSE"
	draw_multiline_string(ThemeDB.fallback_font, Vector2(28, 676), footer, HORIZONTAL_ALIGNMENT_LEFT, 930, 10, 12, Color("#8299a3"))
	if message_time > 0:
		draw_string(ThemeDB.fallback_font, Vector2(650, 42), message, HORIZONTAL_ALIGNMENT_RIGHT, 300, 11, Color("#f3cb68"))


func _system_color(key: String) -> Color:
	return {"neural": Color("#d978ff"), "circulation": Color("#ff4d78"), "respiration": Color("#6ce8ff"), "digestion": Color("#ffc45a"), "senses": Color("#f4ffff")}.get(key, Color.WHITE)


func _draw_meter(position: Vector2, label: String, value: float, color: Color) -> void:
	draw_string(ThemeDB.fallback_font, position, label, HORIZONTAL_ALIGNMENT_LEFT, 120, 9, Color("#9ab0b9"))
	draw_string(ThemeDB.fallback_font, position + Vector2(205, 0), "%3d%%" % roundi(value * 100), HORIZONTAL_ALIGNMENT_RIGHT, 55, 9, color)
	draw_rect(Rect2(position + Vector2(0, 9), Vector2(260, 7)), Color("#13242c"), true)
	draw_rect(Rect2(position + Vector2(0, 9), Vector2(260 * clampf(value, 0, 1), 7)), color, true)


func draw_ellipse(center: Vector2, radius: Vector2, color: Color) -> void:
	var points := PackedVector2Array()
	for index in range(32):
		var angle := TAU * float(index) / 32.0
		points.append(center + Vector2(cos(angle) * radius.x, sin(angle) * radius.y))
	draw_colored_polygon(points, color)
