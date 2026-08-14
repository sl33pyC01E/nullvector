extends "res://scripts/cellular_organism_lab.gd"

const MOTION_FORMAT := "nullvector-cellular-neuromuscular-native-catalog-v4"
const EXPECTED_MOTIONS := ["idle_breathe", "idle_wiggle", "locomote", "joy", "anger", "fear", "confused", "sleep", "taunt", "attack", "cast", "hit", "death"]
const EXPECTED_FACINGS := ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
const EXPECTED_DRIVERS := ["body_bob", "body_sway", "body_squash", "head_tilt", "appendage_left", "appendage_right", "locomotor_left", "locomotor_right", "auxiliary", "weapon_recoil", "sensory_focus", "emission_pulse", "propulsion", "pain_spasm"]
const PHYSIOLOGY_FORMAT := "nullvector-connected-cellular-physiology-native-catalog-v3"
const PHYSIOLOGY_RUNTIME_FORMAT := "nullvector-connected-cellular-physiology-runtime-v1"
const SYSTEM_NAMES := ["circulation", "respiration", "digestion", "neural", "sensory", "locomotion", "reproduction", "immune"]

@export_file("*.json") var motion_catalog_path := "res://generated/cellular_motion/v4/motion_catalog.json"
@export_file("*.json") var physiology_catalog_path := "res://generated/cellular_physiology/v3/catalog.json"
@export_dir var physiology_asset_root := "res://generated/cellular_physiology/v3/"

var motion_catalog: Dictionary = {}
var motion_identities: Dictionary = {}
var physiology_catalog: Dictionary = {}
var physiology_identities: Dictionary = {}
var selected_motion := 0
var selected_facing := 0
var motion_epoch := 0.0
var last_event_frame := -1
var motion_label: Label
var driver_label: Label


func _ready() -> void:
	motion_catalog = _load_json(motion_catalog_path)
	physiology_catalog = _load_json(physiology_catalog_path)
	_validate_motion_catalog()
	_validate_physiology_catalog()
	super()
	_cross_validate_motion_identities()
	_build_motion_overlay()
	if not startup_errors.is_empty():
		status_label.text = "FAIL-CLOSED // " + ", ".join(startup_errors)
		status_label.modulate = Color("#ff526d")
	_refresh_motion_overlay()
	if "--cellular-motion-smoke" in OS.get_cmdline_user_args():
		call_deferred("_run_motion_smoke")


func _validate_physiology_catalog() -> void:
	if physiology_catalog.get("format", "") != PHYSIOLOGY_FORMAT: startup_errors.append("physiology format")
	if physiology_catalog.get("status", "") != "ready": startup_errors.append("physiology status")
	if int(physiology_catalog.get("identity_count", -1)) != 45 or int(physiology_catalog.get("system_count", -1)) != 8: startup_errors.append("physiology census")
	if physiology_catalog.get("systems", []) != SYSTEM_NAMES: startup_errors.append("physiology vocabulary")
	for identity in physiology_catalog.get("identities", []): physiology_identities[str(identity.get("sample_id", ""))] = identity
	if physiology_identities.size() != 45: startup_errors.append("physiology identities")


func _load_physiology_data(sample_id: String) -> Dictionary:
	var identity: Dictionary = physiology_identities.get(sample_id, {}); var artifact: Dictionary = identity.get("runtime", {})
	var path := physiology_asset_root + str(artifact.get("path", ""))
	if not FileAccess.file_exists(path) or FileAccess.get_file_as_bytes(path).size() != int(artifact.get("bytes", -1)) or FileAccess.get_sha256(path) != str(artifact.get("sha256", "")): return {}
	var data := _load_json(path)
	if data.get("format", "") != PHYSIOLOGY_RUNTIME_FORMAT or str(data.get("sample_id", "")) != sample_id: return {}
	return data


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
	var physiology := _load_physiology_data(str(data.get("sample_id", "")))
	if physiology.is_empty() or int(physiology.get("physical_cell_count", -1)) != organism["position"].size():
		startup_errors.append("physiology identity " + str(data.get("sample_id", "?"))); return organism
	organism["physiology_role"] = physiology.get("system_role", []).duplicate(true)
	organism["physiology_weight"] = physiology.get("system_weight", []).duplicate(true)
	organism["physiology_systems"] = physiology.get("systems", []).duplicate(true)
	organism["physiology_capacities"] = {"circulation": 1.0, "respiration": 1.0, "digestion": 1.0, "neural": 1.0, "sensory": 1.0, "locomotion": 1.0, "reproduction": 1.0, "immune": 1.0}
	organism["physiology_oxygen"] = 1.0; organism["physiology_clock"] = 0.0
	organism["physiology_base_digestion"] = float(organism["genome"].get("digestion_efficiency", 0.7))
	organism["physiology_base_regeneration"] = float(organism["genome"].get("tissue_regeneration_rate", 0.01))
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


func _physiology_reachable(organism: Dictionary, role_row: Array) -> Dictionary:
	var reachable: Dictionary = {}; var queue: Array[int] = []
	for index in range(mini(role_row.size(), organism["alive"].size())):
		if int(role_row[index]) == 1 and organism["alive"][index]: reachable[str(index)] = true; queue.append(index)
	var cursor := 0
	while cursor < queue.size():
		var current := queue[cursor]; cursor += 1
		for edge in organism.get("motion_neighbors", [])[current]:
			var neighbor := int(edge[0]); var bond_index := int(edge[1])
			if organism["bond_alive"][bond_index] and organism["alive"][neighbor] and not reachable.has(str(neighbor)):
				reachable[str(neighbor)] = true; queue.append(neighbor)
	return reachable


func _compute_physiology_capacities(organism: Dictionary) -> Dictionary:
	var roles: Array = organism.get("physiology_role", []); var weights: Array = organism.get("physiology_weight", [])
	if roles.size() != SYSTEM_NAMES.size() or weights.size() != SYSTEM_NAMES.size(): return {}
	var raw: Dictionary = {}
	for system_id in range(SYSTEM_NAMES.size()):
		var reachable := _physiology_reachable(organism, roles[system_id]); var total := 0.0; var surviving := 0.0; var connected := 0.0; var core_total := 0.0; var core_alive := 0.0
		for index in range(organism["alive"].size()):
			var weight := float(weights[system_id][index]); var role := int(roles[system_id][index])
			if weight <= 0.0: continue
			total += weight
			if role == 1: core_total += weight
			if organism["alive"][index]:
				surviving += weight
				if reachable.has(str(index)): connected += weight
				if role == 1: core_alive += weight
		var core_fraction := core_alive / maxf(0.0001, core_total); var survival := surviving / maxf(0.0001, total); var connected_fraction := connected / maxf(0.0001, total)
		raw[SYSTEM_NAMES[system_id]] = clampf(pow(core_fraction, 1.55) * (0.25 * survival + 0.75 * connected_fraction), 0.0, 1.0)
	var result: Dictionary = {}
	result["circulation"] = float(raw.get("circulation", 0.0))
	result["respiration"] = float(raw.get("respiration", 0.0)) * pow(float(result["circulation"]), 0.45)
	result["digestion"] = float(raw.get("digestion", 0.0)) * pow(float(result["circulation"]), 0.45)
	result["neural"] = float(raw.get("neural", 0.0)) * pow(float(result["circulation"]), 0.45) * pow(float(result["respiration"]), 0.45)
	result["sensory"] = float(raw.get("sensory", 0.0)) * pow(float(result["neural"]), 0.45)
	result["locomotion"] = float(raw.get("locomotion", 0.0)) * pow(float(result["neural"]), 0.45) * pow(float(result["circulation"]), 0.45) * pow(float(result["respiration"]), 0.45)
	result["reproduction"] = float(raw.get("reproduction", 0.0)) * pow(float(result["circulation"]), 0.45) * pow(float(result["respiration"]), 0.45) * pow(float(result["digestion"]), 0.45)
	result["immune"] = float(raw.get("immune", 0.0)) * pow(float(result["circulation"]), 0.45) * pow(float(result["digestion"]), 0.45)
	return result


func _prepare_physiology(organism: Dictionary, delta: float) -> void:
	organism["physiology_clock"] = float(organism.get("physiology_clock", 0.0)) - delta
	if float(organism["physiology_clock"]) <= 0.0:
		var capacities := _compute_physiology_capacities(organism)
		if not capacities.is_empty(): organism["physiology_capacities"] = capacities
		organism["physiology_clock"] = 0.10
	var capacity: Dictionary = organism.get("physiology_capacities", {})
	organism["genome"]["digestion_efficiency"] = float(organism.get("physiology_base_digestion", 0.7)) * float(capacity.get("digestion", 0.0))
	organism["genome"]["tissue_regeneration_rate"] = float(organism.get("physiology_base_regeneration", 0.01)) * float(capacity.get("immune", 0.0)) * float(capacity.get("circulation", 0.0))


func _advance_physiology(organism: Dictionary, delta: float) -> void:
	var capacity: Dictionary = organism.get("physiology_capacities", {}); var oxygen := float(organism.get("physiology_oxygen", 1.0))
	oxygen += 0.65 * float(capacity.get("respiration", 0.0)) * float(capacity.get("circulation", 0.0)) * delta
	oxygen -= (0.08 + 0.12 * float(capacity.get("locomotion", 0.0))) * delta; oxygen = clampf(oxygen, 0.0, 1.0); organism["physiology_oxygen"] = oxygen
	if oxygen < 0.14:
		var roles: Array = organism.get("physiology_role", []); var neural_row: Array = roles[3] if roles.size() > 3 else []
		for index in range(mini(neural_row.size(), organism["alive"].size())):
			if organism["alive"][index] and int(neural_row[index]) > 0:
				organism["health"][index] = float(organism["health"][index]) - (0.14 - oxygen) * 0.7 * delta
				if float(organism["health"][index]) <= 0.0: organism["alive"][index] = false; organism["health"][index] = 0.0
	if float(capacity.get("circulation", 0.0)) < 0.08:
		for index in range(organism["alive"].size()):
			if organism["alive"][index]: organism["health"][index] = float(organism["health"][index]) - 0.035 * delta


func _apply_motion_force(organism: Dictionary, delta: float) -> void:
	var reachable := _neural_reachable_cells(organism)
	var channel_integrity := _channel_integrities(organism, reachable)
	var physiology: Dictionary = organism.get("physiology_capacities", {})
	var neural_integrity := float(channel_integrity.get("neural", 0.0)) * float(physiology.get("neural", 1.0))
	var motor_integrity := neural_integrity * float(physiology.get("locomotion", 1.0)); organism["motion_neural_integrity"] = neural_integrity
	if neural_integrity <= 0.08: return
	var state := _current_frame(organism)
	if state.is_empty(): return
	var drivers := _driver_map(state["frame"]); var center := _organism_center(organism)
	var facing_angle := deg_to_rad(float(state["facing"].get("rotation_degrees", 0.0)))
	var facing_vector := Vector2(0, -1).rotated(facing_angle)
	var health_fraction := _sum_float(organism["health"]) / maxf(0.001, _sum_float(organism["max_health"]))
	var neural_gain := smoothstep(0.08, 0.72, motor_integrity)
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
	_prepare_physiology(organism, delta)
	_apply_motion_force(organism, delta)
	super(organism, delta)
	_advance_physiology(organism, delta)


func _can_reproduce(organism: Dictionary) -> bool:
	return float(organism.get("physiology_capacities", {}).get("reproduction", 0.0)) >= 0.62 and super(organism)


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
	if organisms.is_empty(): driver_label.text = "W/S MOTION  ARROWS FACING  //  ORGAN TARGETS + LIVE SPRINGS"; return
	var capacity: Dictionary = organisms[0].get("physiology_capacities", {})
	driver_label.text = "BRAIN %3d%%  HEART %3d%%  LUNG %3d%%  GUT %3d%%  O2 %3d%%" % [int(float(capacity.get("neural", 0.0)) * 100.0), int(float(capacity.get("circulation", 0.0)) * 100.0), int(float(capacity.get("respiration", 0.0)) * 100.0), int(float(capacity.get("digestion", 0.0)) * 100.0), int(float(organisms[0].get("physiology_oxygen", 0.0)) * 100.0)]


func _refresh_labels() -> void:
	super()
	_refresh_motion_overlay()


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
	var actuation_velocity := 0.0; var trauma := {"killed": 0, "bonds": 0}; var population := organisms.size(); var physiology_core_damage_verified := false
	if not organisms.is_empty():
		var baseline_capacity := _compute_physiology_capacities(organisms[0])
		if baseline_capacity.size() != 8 or baseline_capacity.values().any(func(value): return float(value) < 0.99): errors.append("physiology baseline")
		selected_motion = EXPECTED_MOTIONS.find("locomote"); selected_facing = 2; motion_epoch = 0.0; simulation_time = 0.25
		_apply_motion_force(organisms[0], 1.0 / 60.0)
		for velocity in organisms[0]["velocity"]: actuation_velocity += velocity.length()
		if not is_finite(actuation_velocity) or actuation_velocity <= 0.01: errors.append("neuromuscular actuation")
		if _feed_current(20.0) <= 0.0: errors.append("motion body feeding")
		if not _reproduce(organisms[0], true): errors.append("motion body reproduction")
		trauma = _damage_at(_organism_center(organisms[0]), 38.0, 2.2, 260.0)
		if int(trauma["killed"]) <= 0 or int(trauma["bonds"]) <= 0: errors.append("motion body damage")
		population = organisms.size()
		if organisms.size() >= 2:
			var diagnostic: Dictionary = organisms[1]; var neural_roles: Array = diagnostic.get("physiology_role", [])[3]
			for index in range(neural_roles.size()):
				if int(neural_roles[index]) == 1: diagnostic["alive"][index] = false; diagnostic["health"][index] = 0.0
			for bond_index in range(diagnostic["bond_ab"].size()):
				var pair: Array = diagnostic["bond_ab"][bond_index]
				if not diagnostic["alive"][int(pair[0])] or not diagnostic["alive"][int(pair[1])]: _break_bond(diagnostic, bond_index)
			var damaged_capacity := _compute_physiology_capacities(diagnostic)
			physiology_core_damage_verified = float(damaged_capacity.get("neural", 1.0)) == 0.0 and float(damaged_capacity.get("locomotion", 1.0)) == 0.0 and float(damaged_capacity.get("circulation", 0.0)) > 0.9
			if not physiology_core_damage_verified: errors.append("physiology brain cascade")
	var report := {"format": "nullvector-cellular-motion-godot-smoke-v2", "passed": errors.is_empty(), "errors": errors, "engine": Engine.get_version_info().get("string", ""), "motion_bundle_id": motion_catalog.get("bundle_id", ""), "physiology_bundle_id": physiology_catalog.get("bundle_id", ""), "organism_bundle_id": catalog.get("bundle_id", ""), "identity_count": motion_catalog.get("identity_count", 0), "physiology_identity_count": physiology_catalog.get("identity_count", 0), "physiology_system_count": physiology_catalog.get("system_count", 0), "physiology_core_damage_verified": physiology_core_damage_verified, "clip_count": clip_count, "frame_count": frame_count, "event_count": event_count, "mapped_organs": mapped_organs, "actuation_velocity": actuation_velocity, "damage_killed": trauma["killed"], "damage_bonds": trauma["bonds"], "population_after_reproduction": population, "python_runtime_required": false}
	if not report_path.is_empty():
		var file := FileAccess.open(report_path, FileAccess.WRITE)
		if file != null: file.store_string(JSON.stringify(report, "  ", true) + "\n")
	if errors.is_empty(): print("CELLULAR_MOTION_SMOKE_OK identities=45 clips=%d frames=%d organs=%d velocity=%.3f population=%d" % [clip_count, frame_count, mapped_organs, actuation_velocity, population])
	else: push_error("CELLULAR_MOTION_SMOKE_FAIL " + ", ".join(errors))
	get_tree().quit(0 if errors.is_empty() else 1)
