extends "res://scripts/cellular_organism_lab.gd"

const MOTION_FORMAT := "nullvector-cellular-neuromuscular-native-catalog-v2"
const EXPECTED_MOTIONS := ["idle_breathe", "idle_wiggle", "locomote", "joy", "anger", "fear", "confused", "sleep", "taunt", "attack", "cast", "hit", "death"]
const EXPECTED_FACINGS := ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
const EXPECTED_DRIVERS := ["body_bob", "body_sway", "body_squash", "head_tilt", "appendage_left", "appendage_right", "locomotor_left", "locomotor_right", "auxiliary", "weapon_recoil", "sensory_focus", "emission_pulse", "propulsion", "pain_spasm"]

@export_file("*.json") var motion_catalog_path := "res://generated/cellular_motion/v2/motion_catalog.json"

var motion_catalog: Dictionary = {}
var motion_identities: Dictionary = {}
var selected_motion := 0
var selected_facing := 0
var motion_epoch := 0.0
var last_event_frame := -1
var motion_label: Label
var driver_label: Label


func _ready() -> void:
	motion_catalog = _load_json(motion_catalog_path)
	_validate_motion_catalog()
	super()
	_cross_validate_motion_identities()
	_build_motion_overlay()
	if not startup_errors.is_empty():
		status_label.text = "FAIL-CLOSED // " + ", ".join(startup_errors)
		status_label.modulate = Color("#ff526d")
	_refresh_motion_overlay()
	if "--cellular-motion-smoke" in OS.get_cmdline_user_args():
		call_deferred("_run_motion_smoke")


func _validate_motion_catalog() -> void:
	if motion_catalog.get("format", "") != MOTION_FORMAT: startup_errors.append("motion format")
	if motion_catalog.get("status", "") != "ready": startup_errors.append("motion status")
	for pair in [["identity_count", 45], ["family_count", 5], ["motion_count", 13], ["facing_count", 8], ["clip_count", 520], ["frame_count", 4720]]:
		if int(motion_catalog.get(pair[0], -1)) != int(pair[1]): startup_errors.append("motion " + str(pair[0]))
	if motion_catalog.get("motions", []) != EXPECTED_MOTIONS: startup_errors.append("motion vocabulary")
	if motion_catalog.get("facings", []) != EXPECTED_FACINGS: startup_errors.append("facing vocabulary")
	if motion_catalog.get("drivers", []) != EXPECTED_DRIVERS: startup_errors.append("driver vocabulary")
	if motion_catalog.get("programs", []).size() != 5: startup_errors.append("family program census")
	for identity in motion_catalog.get("identities", []):
		motion_identities[str(identity.get("sample_id", ""))] = identity
	if motion_identities.size() != 45: startup_errors.append("motion identity census")


func _cross_validate_motion_identities() -> void:
	if catalog.is_empty(): return
	for entry in catalog.get("species", []):
		var sample_id := str(entry.get("sample_id", "")); var identity: Dictionary = motion_identities.get(sample_id, {})
		if identity.is_empty(): startup_errors.append("motion identity " + sample_id); continue
		if int(identity.get("family_id", -1)) != int(entry.get("family_id", -2)): startup_errors.append("motion family " + sample_id)
		if int(identity.get("physical_cell_count", -1)) != int(entry.get("summary", {}).get("physical_cell_count", -2)): startup_errors.append("motion cells " + sample_id)


func _build_motion_overlay() -> void:
	var canvas := CanvasLayer.new(); add_child(canvas)
	var panel := _panel(canvas, Rect2(850, 98, 384, 70))
	motion_label = _label(panel, Vector2(12, 8), Vector2(360, 22), "MOTION", CYAN, 10)
	driver_label = _label(panel, Vector2(12, 32), Vector2(360, 28), "DRIVERS", MUTED, 8)
	controls_label.text += "\nW/S motion  Arrows facing"


func _create_organism(data: Dictionary, center: Vector2, generation: int, mutation_seed: int) -> Dictionary:
	var organism: Dictionary = super(data, center, generation, mutation_seed)
	if organism.is_empty(): return organism
	var rest_local: Array = []; var rest_center := Vector2.ZERO
	for position in organism["position"]: rest_center += position
	rest_center /= maxi(1, organism["position"].size())
	for position in organism["position"]: rest_local.append(position - rest_center)
	var identity: Dictionary = motion_identities.get(str(data.get("sample_id", "")), {})
	var channel_by_organ: Dictionary = {}
	for channel in identity.get("channels", {}):
		for organ_id in identity.get("channels", {}).get(channel, []): channel_by_organ[str(int(organ_id))] = str(channel)
	var attachment_by_organ: Dictionary = {}
	for attachment in identity.get("attachments", []):
		var root_cell := int(attachment.get("root_cell", -1)); var organ_key := str(int(attachment.get("organ_id", -1)))
		if root_cell < 0 or root_cell >= rest_local.size(): continue
		attachment_by_organ[organ_key] = {"root_cell": root_cell, "pivot": rest_local[root_cell], "parent_organ_id": int(attachment.get("parent_organ_id", 0)), "maximum_radius": float(attachment.get("maximum_radius", 0.0))}
	var channel_initial_cells: Dictionary = {}
	for index in range(organism["organ_id"].size()):
		var channel := str(channel_by_organ.get(str(int(organism["organ_id"][index])), "chassis"))
		channel_initial_cells[channel] = int(channel_initial_cells.get(channel, 0)) + 1
	var motion_neighbors: Array = []
	for index in range(organism["position"].size()): motion_neighbors.append([])
	for bond_index in range(organism["bond_ab"].size()):
		var pair: Array = organism["bond_ab"][bond_index]; var a := int(pair[0]); var b := int(pair[1])
		motion_neighbors[a].append([b, bond_index]); motion_neighbors[b].append([a, bond_index])
	organism["motion_rest_local"] = rest_local
	organism["motion_attachment_by_organ"] = attachment_by_organ
	organism["motion_channel_by_organ"] = channel_by_organ
	organism["motion_channel_initial_cells"] = channel_initial_cells
	organism["motion_neighbors"] = motion_neighbors
	organism["motion_emission_pulse"] = 0.0
	organism["motion_energy_spent"] = 0.0
	organism["motion_neural_integrity"] = 1.0
	return organism


func _current_clip(family_id: int) -> Dictionary:
	var programs: Array = motion_catalog.get("programs", [])
	if family_id < 0 or family_id >= programs.size(): return {}
	var clips: Array = programs[family_id].get("clips", [])
	return clips[selected_motion] if selected_motion >= 0 and selected_motion < clips.size() else {}


func _current_frame(organism: Dictionary) -> Dictionary:
	var clip := _current_clip(int(organism["data"].get("family_id", 0)))
	if clip.is_empty(): return {}
	var facings: Array = clip.get("facings", []); var facing: Dictionary = facings[selected_facing]
	var frames: Array = facing.get("frames", []); var elapsed := maxf(0.0, simulation_time - motion_epoch)
	var playback_count: int = maxi(1, int(clip.get("frame_count", 1)) - (1 if bool(clip.get("loop", false)) else 0))
	var frame_index := int(floor(elapsed * float(clip.get("fps", 1))))
	frame_index = posmod(frame_index, playback_count) if bool(clip.get("loop", false)) else mini(playback_count - 1, frame_index)
	return {"clip": clip, "facing": facing, "frame": frames[frame_index], "frame_index": frame_index}


func _driver_map(frame: Dictionary) -> Dictionary:
	var result: Dictionary = {}; var values: Array = frame.get("drivers", [])
	for index in range(mini(values.size(), EXPECTED_DRIVERS.size())): result[EXPECTED_DRIVERS[index]] = float(values[index])
	return result


func _channel_driver(channel: String, drivers: Dictionary) -> float:
	match channel:
		"neural": return float(drivers.get("head_tilt", 0.0))
		"sensory": return float(drivers.get("head_tilt", 0.0)) + float(drivers.get("sensory_focus", 0.0)) * 0.18
		"left_appendage": return float(drivers.get("appendage_left", 0.0))
		"right_appendage": return float(drivers.get("appendage_right", 0.0))
		"left_locomotor": return float(drivers.get("locomotor_left", 0.0))
		"right_locomotor": return float(drivers.get("locomotor_right", 0.0))
		"auxiliary": return float(drivers.get("auxiliary", 0.0))
		"weapon": return float(drivers.get("weapon_recoil", 0.0))
		"emitter": return float(drivers.get("auxiliary", 0.0)) * 0.5
	return 0.0


func _neural_reachable_cells(organism: Dictionary) -> Dictionary:
	var channels: Dictionary = organism.get("motion_channel_by_organ", {}); var reachable: Dictionary = {}; var queue: Array[int] = []
	for index in range(organism["alive"].size()):
		if organism["alive"][index] and str(channels.get(str(int(organism["organ_id"][index])), "")) == "neural":
			reachable[str(index)] = true; queue.append(index)
	var cursor := 0
	while cursor < queue.size():
		var current := queue[cursor]; cursor += 1
		for edge in organism.get("motion_neighbors", [])[current]:
			var neighbor := int(edge[0]); var bond_index := int(edge[1])
			if organism["bond_alive"][bond_index] and organism["alive"][neighbor] and not reachable.has(str(neighbor)):
				reachable[str(neighbor)] = true; queue.append(neighbor)
	return reachable


func _channel_integrities(organism: Dictionary, reachable: Dictionary) -> Dictionary:
	var initial: Dictionary = organism.get("motion_channel_initial_cells", {}); var functioning: Dictionary = {}; var channels: Dictionary = organism.get("motion_channel_by_organ", {})
	for index in range(organism["alive"].size()):
		if organism["alive"][index] and reachable.has(str(index)):
			var channel := str(channels.get(str(int(organism["organ_id"][index])), "chassis")); functioning[channel] = int(functioning.get(channel, 0)) + 1
	var result: Dictionary = {}
	for channel in initial:
		var count := int(initial[channel]); result[channel] = clampf(float(functioning.get(channel, 0)) / maxf(1.0, float(count)), 0.0, 1.0)
	return result


func _apply_motion_force(organism: Dictionary, delta: float) -> void:
	var reachable := _neural_reachable_cells(organism)
	var channel_integrity := _channel_integrities(organism, reachable)
	var neural_integrity := float(channel_integrity.get("neural", 0.0)); organism["motion_neural_integrity"] = neural_integrity
	if neural_integrity <= 0.08: return
	var state := _current_frame(organism)
	if state.is_empty(): return
	var drivers := _driver_map(state["frame"]); var center := _organism_center(organism)
	var facing_angle := deg_to_rad(float(state["facing"].get("rotation_degrees", 0.0)))
	var facing_vector := Vector2(0, -1).rotated(facing_angle)
	var health_fraction := _sum_float(organism["health"]) / maxf(0.001, _sum_float(organism["max_health"]))
	var neural_gain := smoothstep(0.08, 0.72, neural_integrity)
	var strength: float = clampf(health_fraction, 0.08, 1.0) * 3.05 * neural_gain
	var rest_local: Array = organism["motion_rest_local"]; var attachments: Dictionary = organism["motion_attachment_by_organ"]; var channels: Dictionary = organism["motion_channel_by_organ"]
	var squash := float(drivers.get("body_squash", 0.0)); var work := 0.0
	for index in range(organism["position"].size()):
		if not organism["alive"][index] or not reachable.has(str(index)): continue
		var local: Vector2 = rest_local[index]
		local.x *= 1.0 + squash * 0.10; local.y *= 1.0 - squash * 0.08
		local += Vector2(float(drivers.get("body_sway", 0.0)) * 6.5, float(drivers.get("body_bob", 0.0)) * 8.0)
		var organ_key := str(int(organism["organ_id"][index])); var channel := str(channels.get(organ_key, "chassis")); var amount := _channel_driver(channel, drivers)
		var channel_gain := float(channel_integrity.get(channel, 1.0))
		if absf(amount) > 0.00001 and channel != "chassis" and attachments.has(organ_key):
			var attachment: Dictionary = attachments[organ_key]; var root_cell := int(attachment.get("root_cell", -1))
			if root_cell < 0 or not organism["alive"][root_cell] or not reachable.has(str(root_cell)): continue
			var pivot: Vector2 = attachment.get("pivot", Vector2.ZERO); local = pivot + (local - pivot).rotated(amount * channel_gain * deg_to_rad(30.0))
		if neural_integrity < 0.72:
			var tremor := sin(simulation_time * (10.0 + float(index % 7)) + float(index) * 1.618) * (1.0 - neural_integrity)
			local += Vector2(tremor, -tremor * 0.65) * 2.4
		local = local.rotated(facing_angle)
		var target := center + local; var error: Vector2 = target - organism["position"][index]
		if error.length() > 36.0: error = error.normalized() * 36.0
		var impulse: Vector2 = error * strength * delta + facing_vector * float(drivers.get("propulsion", 0.0)) * 18.0 * delta * neural_gain
		organism["velocity"][index] += impulse; work += impulse.length()
	organism["motion_emission_pulse"] = float(drivers.get("emission_pulse", 0.0))
	organism["motion_energy_spent"] = float(organism["motion_energy_spent"]) + work * 0.000015
	var alive_count := maxi(1, organism["alive"].count(true)); var energy_cost := work * 0.000015 / alive_count
	for index in range(organism["energy"].size()):
		if organism["alive"][index]: organism["energy"][index] = maxf(0.0, float(organism["energy"][index]) - energy_cost)
	if int(state["frame_index"]) != last_event_frame:
		last_event_frame = int(state["frame_index"])
		for event in state["clip"].get("events", []):
			if int(event.get("frame", -1)) == last_event_frame: _event("MOTOR EVENT // " + str(event.get("name", "?")).to_upper(), CYAN)


func _step_organism(organism: Dictionary, delta: float) -> void:
	_apply_motion_force(organism, delta)
	super(organism, delta)


func _cell_color(organism: Dictionary, index: int) -> Color:
	var color: Color = super(organism, index)
	if int(organism["emission"][index]) > 0:
		var pulse: float = clampf(float(organism.get("motion_emission_pulse", 0.0)), 0.0, 1.0)
		color = color.lerp(Color.WHITE, pulse * 0.24)
	return color


func _select_motion(delta: int) -> void:
	selected_motion = posmod(selected_motion + delta, EXPECTED_MOTIONS.size()); motion_epoch = simulation_time; last_event_frame = -1
	_event("MOTION // " + str(EXPECTED_MOTIONS[selected_motion]).to_upper(), LIME); _refresh_motion_overlay()


func _select_facing(delta: int) -> void:
	selected_facing = posmod(selected_facing + delta, EXPECTED_FACINGS.size()); motion_epoch = simulation_time; last_event_frame = -1
	_event("FACING // " + str(EXPECTED_FACINGS[selected_facing]).to_upper(), LIME); _refresh_motion_overlay()


func _refresh_motion_overlay() -> void:
	if motion_label == null: return
	var clip := _current_clip(int(organisms[0]["data"].get("family_id", 0))) if not organisms.is_empty() else {}
	motion_label.text = "%s // %s // %s" % [str(EXPECTED_MOTIONS[selected_motion]).to_upper(), str(EXPECTED_FACINGS[selected_facing]).to_upper(), "%d FPS" % int(clip.get("fps", 0))]
	driver_label.text = "W/S MOTION  ARROWS FACING  //  ORGAN TARGETS + LIVE SPRINGS"


func _unhandled_input(event: InputEvent) -> void:
	super(event)
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_W: _select_motion(-1)
			KEY_S: _select_motion(1)
			KEY_LEFT: _select_facing(-1)
			KEY_RIGHT: _select_facing(1)


func _run_motion_smoke() -> void:
	var report_path := ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--cellular-motion-report="): report_path = argument.trim_prefix("--cellular-motion-report=")
	var errors: Array[String] = startup_errors.duplicate(); var clip_count := 0; var frame_count := 0; var event_count := 0; var mapped_organs := 0
	for program in motion_catalog.get("programs", []):
		for clip in program.get("clips", []):
			for facing in clip.get("facings", []):
				clip_count += 1
				for frame in facing.get("frames", []):
					frame_count += 1
					if frame.get("drivers", []).size() != EXPECTED_DRIVERS.size(): errors.append("driver width")
			event_count += clip.get("events", []).size()
	for identity in motion_catalog.get("identities", []):
		var seen: Dictionary = {}
		for channel in identity.get("channels", {}):
			for organ_id in identity.get("channels", {}).get(channel, []):
				var key := str(int(organ_id));
				if seen.has(key): errors.append("duplicate organ mapping")
				seen[key] = true; mapped_organs += 1
		if seen.size() != int(identity.get("organ_count", -1)): errors.append("organ channel census")
	if clip_count != 520 or frame_count != 4720: errors.append("motion totals")
	var actuation_velocity := 0.0; var trauma := {"killed": 0, "bonds": 0}; var population := organisms.size()
	if not organisms.is_empty():
		selected_motion = EXPECTED_MOTIONS.find("locomote"); selected_facing = 2; motion_epoch = 0.0; simulation_time = 0.25
		_apply_motion_force(organisms[0], 1.0 / 60.0)
		for velocity in organisms[0]["velocity"]: actuation_velocity += velocity.length()
		if not is_finite(actuation_velocity) or actuation_velocity <= 0.01: errors.append("neuromuscular actuation")
		trauma = _damage_at(_organism_center(organisms[0]), 38.0, 2.2, 260.0)
		if int(trauma["killed"]) <= 0 or int(trauma["bonds"]) <= 0: errors.append("motion body damage")
		if _feed_current(20.0) <= 0.0: errors.append("motion body feeding")
		if not _reproduce(organisms[0], true): errors.append("motion body reproduction")
		population = organisms.size()
	var report := {"format": "nullvector-cellular-motion-godot-smoke-v1", "passed": errors.is_empty(), "errors": errors, "engine": Engine.get_version_info().get("string", ""), "motion_bundle_id": motion_catalog.get("bundle_id", ""), "organism_bundle_id": catalog.get("bundle_id", ""), "identity_count": motion_catalog.get("identity_count", 0), "clip_count": clip_count, "frame_count": frame_count, "event_count": event_count, "mapped_organs": mapped_organs, "actuation_velocity": actuation_velocity, "damage_killed": trauma["killed"], "damage_bonds": trauma["bonds"], "population_after_reproduction": population, "python_runtime_required": false}
	if not report_path.is_empty():
		var file := FileAccess.open(report_path, FileAccess.WRITE)
		if file != null: file.store_string(JSON.stringify(report, "  ", true) + "\n")
	if errors.is_empty(): print("CELLULAR_MOTION_SMOKE_OK identities=45 clips=%d frames=%d organs=%d velocity=%.3f population=%d" % [clip_count, frame_count, mapped_organs, actuation_velocity, population])
	else: push_error("CELLULAR_MOTION_SMOKE_FAIL " + ", ".join(errors))
	get_tree().quit(0 if errors.is_empty() else 1)
