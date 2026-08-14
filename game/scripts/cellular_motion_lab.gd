extends "res://scripts/cellular_organism_lab.gd"

const MOTION_FORMAT := "nullvector-cellular-neuromuscular-native-catalog-v7"
const EXPECTED_MOTIONS := ["idle_breathe", "idle_wiggle", "locomote", "joy", "anger", "fear", "confused", "sleep", "taunt", "attack", "cast", "hit", "death"]
const EXPECTED_FACINGS := ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
const EXPECTED_DRIVERS := ["body_bob", "body_sway", "body_squash", "head_tilt", "appendage_left", "appendage_right", "locomotor_left", "locomotor_right", "auxiliary", "weapon_recoil", "sensory_focus", "emission_pulse", "propulsion", "pain_spasm"]
const PHYSIOLOGY_FORMAT := "nullvector-connected-cellular-physiology-native-catalog-v6"
const PHYSIOLOGY_RUNTIME_FORMAT := "nullvector-connected-cellular-physiology-runtime-v4"
const TRAUMA_FORMAT := "nullvector-cellular-trauma-native-catalog-v3"
const TRAUMA_RUNTIME_FORMAT := "nullvector-cellular-trauma-runtime-v3"
const SYSTEM_NAMES := ["circulation", "respiration", "digestion", "neural", "sensory", "locomotion", "reproduction", "immune"]
const MOTION_VIEW_NAMES := ["PHENOTYPE", "ORGANS", "FLUID / PRESSURE", "HEALTH", "TISSUE", "SYSTEM NETWORK"]
const SYSTEM_COLORS := [Color("#ff4d67"), Color("#51d9ff"), Color("#ffb347"), Color("#b879ff"), Color("#ffe761"), Color("#69ff91"), Color("#ff70c8"), Color("#70e8c1")]

@export_file("*.json") var motion_catalog_path := "res://generated/cellular_motion/v7/motion_catalog.json"
@export_file("*.json") var physiology_catalog_path := "res://generated/cellular_physiology/v6/catalog.json"
@export_dir var physiology_asset_root := "res://generated/cellular_physiology/v6/"
@export_file("*.json") var trauma_catalog_path := "res://generated/cellular_trauma/v3/catalog.json"
@export_dir var trauma_asset_root := "res://generated/cellular_trauma/v3/"

var motion_catalog: Dictionary = {}
var motion_identities: Dictionary = {}
var physiology_catalog: Dictionary = {}
var physiology_identities: Dictionary = {}
var trauma_catalog: Dictionary = {}
var trauma_identities: Dictionary = {}
var selected_motion := 0
var selected_facing := 0
var selected_system := 0
var motion_epoch := 0.0
var last_event_frame := -1
var motion_label: Label
var driver_label: Label


func _ready() -> void:
	motion_catalog = _load_json(motion_catalog_path)
	physiology_catalog = _load_json(physiology_catalog_path)
	trauma_catalog = _load_json(trauma_catalog_path)
	_validate_motion_catalog()
	_validate_physiology_catalog()
	_validate_trauma_catalog()
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


func _validate_trauma_catalog() -> void:
	if trauma_catalog.get("format", "") != TRAUMA_FORMAT: startup_errors.append("trauma format")
	if trauma_catalog.get("status", "") != "ready": startup_errors.append("trauma status")
	if int(trauma_catalog.get("identity_count", -1)) != 45: startup_errors.append("trauma census")
	if int(trauma_catalog.get("total_cells", -1)) != 25668 or int(trauma_catalog.get("total_bonds", -1)) != 85357: startup_errors.append("trauma totals")
	for identity in trauma_catalog.get("identities", []): trauma_identities[str(identity.get("sample_id", ""))] = identity
	if trauma_identities.size() != 45: startup_errors.append("trauma identities")


func _load_trauma_data(sample_id: String) -> Dictionary:
	var identity: Dictionary = trauma_identities.get(sample_id, {}); var artifact: Dictionary = identity.get("runtime", {})
	var path := trauma_asset_root + str(artifact.get("path", ""))
	if not FileAccess.file_exists(path) or FileAccess.get_file_as_bytes(path).size() != int(artifact.get("bytes", -1)) or FileAccess.get_sha256(path) != str(artifact.get("sha256", "")): return {}
	var data := _load_json(path)
	if data.get("format", "") != TRAUMA_RUNTIME_FORMAT or str(data.get("sample_id", "")) != sample_id: return {}
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
		if not trauma_identities.has(sample_id): startup_errors.append("trauma identity " + sample_id)


func _build_motion_overlay() -> void:
	var canvas := CanvasLayer.new(); add_child(canvas)
	var panel := _panel(canvas, Rect2(850, 98, 384, 70))
	motion_label = _label(panel, Vector2(12, 8), Vector2(360, 22), "MOTION", CYAN, 10)
	driver_label = _label(panel, Vector2(12, 32), Vector2(360, 28), "DRIVERS", MUTED, 8)
	controls_label.text += "\nW/S motion  Arrows facing  C system"


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
	organism["motion_chain_depth"] = _motion_chain_depths(organism, rest_local, attachment_by_organ)
	organism["motion_emission_pulse"] = 0.0
	organism["motion_energy_spent"] = 0.0
	organism["motion_neural_integrity"] = 1.0
	organism["motion_autonomous"] = _uses_autonomous_motion()
	organism["motion_index"] = selected_motion
	organism["motion_facing_index"] = selected_facing
	organism["motion_epoch"] = simulation_time
	organism["motion_last_event_frame"] = -1
	organism["motion_transition_count"] = 0
	organism["motion_event_count"] = 0
	organism["motion_behavior"] = "manual selection"
	organism["motion_lock_until"] = 0.0
	var physiology := _load_physiology_data(str(data.get("sample_id", "")))
	if physiology.is_empty() or int(physiology.get("physical_cell_count", -1)) != organism["position"].size():
		startup_errors.append("physiology identity " + str(data.get("sample_id", "?"))); return organism
	organism["physiology_role"] = physiology.get("system_role", []).duplicate(true)
	organism["physiology_weight"] = physiology.get("system_weight", []).duplicate(true)
	organism["physiology_systems"] = physiology.get("systems", []).duplicate(true)
	organism["physiology_capacities"] = {"circulation": 1.0, "respiration": 1.0, "digestion": 1.0, "neural": 1.0, "sensory": 1.0, "locomotion": 1.0, "reproduction": 1.0, "immune": 1.0}
	organism["physiology_network_reachable"] = []
	organism["physiology_oxygen"] = 1.0; organism["physiology_clock"] = 0.0
	organism["physiology_base_digestion"] = float(organism["genome"].get("digestion_efficiency", 0.7))
	organism["physiology_base_regeneration"] = float(organism["genome"].get("tissue_regeneration_rate", 0.01))
	var trauma := _load_trauma_data(str(data.get("sample_id", "")))
	var trauma_arrays: Dictionary = trauma.get("arrays", {})
	if trauma.is_empty() or trauma_arrays.get("heal_class", []).size() != organism["position"].size() or trauma_arrays.get("bond_repair_weight", []).size() != organism["bond_ab"].size():
		startup_errors.append("trauma identity " + str(data.get("sample_id", "?"))); return organism
	organism["trauma_profile"] = trauma.get("profile", {}).duplicate(true)
	organism["trauma_heal_class"] = trauma_arrays.get("heal_class", []).duplicate()
	organism["trauma_clotting_weight"] = trauma_arrays.get("clotting_weight", []).duplicate()
	organism["trauma_scar_bias"] = trauma_arrays.get("scar_bias", []).duplicate()
	organism["trauma_regrowth_weight"] = trauma_arrays.get("regrowth_weight", []).duplicate()
	organism["trauma_bond_repair_weight"] = trauma_arrays.get("bond_repair_weight", []).duplicate()
	organism["trauma_bond_magnetic_weight"] = trauma_arrays.get("bond_magnetic_weight", []).duplicate()
	organism["trauma_clot"] = _filled_float_array(organism["position"].size(), 0.0)
	organism["trauma_scar"] = _filled_float_array(organism["position"].size(), 0.0)
	organism["trauma_wound_age"] = _filled_float_array(organism["position"].size(), 0.0)
	organism["trauma_bond_age"] = _filled_float_array(organism["bond_ab"].size(), 0.0)
	organism["trauma_fragment_fate"] = _filled_string_array(organism["position"].size(), "attached")
	organism["trauma_component_age"] = {}
	organism["trauma_reconnections"] = 0
	organism["trauma_polyps"] = 0
	organism["trauma_biomass_components"] = 0
	return organism


func _motion_chain_depths(organism: Dictionary, rest_local: Array, attachments: Dictionary) -> Array:
	# A continuous root-to-tip coordinate lets every organ bend progressively.
	# It preserves the authored silhouettes and driver curves while avoiding the
	# old rigid-cardboard rotation shared by every cell in an appendage.
	var result := _filled_float_array(rest_local.size(), 0.0)
	for organ_key in attachments:
		var attachment: Dictionary = attachments[organ_key]
		var pivot: Vector2 = attachment.get("pivot", Vector2.ZERO)
		var radius := maxf(0.001, float(attachment.get("maximum_radius", 0.0)))
		var organ_id := int(organ_key)
		for index in range(rest_local.size()):
			if int(organism["organ_id"][index]) != organ_id: continue
			result[index] = clampf((rest_local[index] as Vector2).distance_to(pivot) / radius, 0.0, 1.0)
	return result


func _filled_float_array(count: int, value: float) -> Array:
	var result: Array = []
	for index in range(count): result.append(value)
	return result


func _filled_string_array(count: int, value: String) -> Array:
	var result: Array = []
	for index in range(count): result.append(value)
	return result


func _uses_autonomous_motion() -> bool:
	return false


func _clip_for(family_id: int, motion_index: int) -> Dictionary:
	var programs: Array = motion_catalog.get("programs", [])
	if family_id < 0 or family_id >= programs.size(): return {}
	var clips: Array = programs[family_id].get("clips", [])
	return clips[motion_index] if motion_index >= 0 and motion_index < clips.size() else {}


func _current_clip(family_id: int) -> Dictionary:
	return _clip_for(family_id, selected_motion)


func _facing_index_from_vector(direction: Vector2) -> int:
	if direction.length_squared() <= 0.000001: return 0
	return posmod(int(round(Vector2.UP.angle_to(direction.normalized()) / (PI * 0.25))), EXPECTED_FACINGS.size())


func _set_organism_motion(organism: Dictionary, motion_name: String, facing: Vector2 = Vector2.ZERO, behavior: String = "", hold_seconds: float = 0.0) -> void:
	var motion_index := EXPECTED_MOTIONS.find(motion_name)
	if motion_index < 0: return
	var facing_index := int(organism.get("motion_facing_index", selected_facing))
	if facing.length_squared() > 0.000001: facing_index = _facing_index_from_vector(facing)
	var changed := motion_index != int(organism.get("motion_index", selected_motion)) or facing_index != int(organism.get("motion_facing_index", selected_facing))
	if changed:
		organism["motion_index"] = motion_index
		organism["motion_facing_index"] = facing_index
		organism["motion_epoch"] = simulation_time
		organism["motion_last_event_frame"] = -1
		organism["motion_transition_count"] = int(organism.get("motion_transition_count", 0)) + 1
		organism["motion_lock_until"] = simulation_time + maxf(0.0, hold_seconds)
	if not behavior.is_empty(): organism["motion_behavior"] = behavior


func _current_frame(organism: Dictionary) -> Dictionary:
	var motion_index := int(organism.get("motion_index", selected_motion)) if bool(organism.get("motion_autonomous", false)) else selected_motion
	var facing_index := int(organism.get("motion_facing_index", selected_facing)) if bool(organism.get("motion_autonomous", false)) else selected_facing
	var epoch := float(organism.get("motion_epoch", motion_epoch)) if bool(organism.get("motion_autonomous", false)) else motion_epoch
	var clip := _clip_for(int(organism["data"].get("family_id", 0)), motion_index)
	if clip.is_empty(): return {}
	var facings: Array = clip.get("facings", [])
	if facing_index < 0 or facing_index >= facings.size(): return {}
	var facing: Dictionary = facings[facing_index]
	var frames: Array = facing.get("frames", []); var elapsed := maxf(0.0, simulation_time - epoch)
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


func _physiology_cell_viability(organism: Dictionary, index: int, system_id: int) -> float:
	if not organism["alive"][index]: return 0.0
	var viability := clampf(float(organism["health"][index]) / maxf(0.0001, float(organism["max_health"][index])), 0.0, 1.0)
	if system_id == 0:
		var baseline: Array = organism.get("fluid_baseline", []); var fluid: Array = organism.get("fluid", [])
		if index < baseline.size() and index < fluid.size() and float(baseline[index]) > 0.000001:
			viability = minf(viability, clampf(float(fluid[index]) / float(baseline[index]), 0.0, 1.0))
	return viability


func _physiology_reachable(organism: Dictionary, role_row: Array, system_id := -1) -> Dictionary:
	# Values are graded widest-path delivery, not booleans. A wounded conduit can
	# remain connected while transmitting less service to its downstream cells.
	var reachable: Dictionary = {}; var queue: Array[int] = []
	for index in range(mini(role_row.size(), organism["alive"].size())):
		if int(role_row[index]) != 1 or not organism["alive"][index]: continue
		var viability := _physiology_cell_viability(organism, index, system_id)
		if viability > 0.0: reachable[str(index)] = viability; queue.append(index)
	var cursor := 0
	while cursor < queue.size():
		var current := queue[cursor]; cursor += 1; var current_signal := float(reachable.get(str(current), 0.0))
		for edge in organism.get("motion_neighbors", [])[current]:
			var neighbor := int(edge[0]); var bond_index := int(edge[1])
			if not organism["bond_alive"][bond_index] or not organism["alive"][neighbor] or int(role_row[neighbor]) <= 0: continue
			var viability := _physiology_cell_viability(organism, neighbor, system_id)
			var candidate := minf(current_signal, viability); var key := str(neighbor)
			if candidate > float(reachable.get(key, 0.0)) + 0.0000001:
				reachable[key] = candidate; queue.append(neighbor)
	return reachable


func _physiology_networks(organism: Dictionary) -> Array:
	var result: Array = []
	var rows: Array = organism.get("physiology_role", [])
	for system_id in range(rows.size()): result.append(_physiology_reachable(organism, rows[system_id], system_id))
	return result


func _compute_physiology_capacities(organism: Dictionary) -> Dictionary:
	var roles: Array = organism.get("physiology_role", []); var weights: Array = organism.get("physiology_weight", [])
	if roles.size() != SYSTEM_NAMES.size() or weights.size() != SYSTEM_NAMES.size(): return {}
	var raw: Dictionary = {}
	for system_id in range(SYSTEM_NAMES.size()):
		var reachable := _physiology_reachable(organism, roles[system_id], system_id); var total := 0.0; var surviving := 0.0; var connected := 0.0; var core_total := 0.0; var core_alive := 0.0
		for index in range(organism["alive"].size()):
			var weight := float(weights[system_id][index]); var role := int(roles[system_id][index])
			if weight <= 0.0: continue
			total += weight
			if role == 1: core_total += weight
			if organism["alive"][index]:
				var viability := _physiology_cell_viability(organism, index, system_id)
				surviving += weight * viability
				connected += weight * float(reachable.get(str(index), 0.0))
				if role == 1: core_alive += weight * viability
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


func _diagnostic_system_core_failures(organism: Dictionary) -> Dictionary:
	var failures: Dictionary = {}
	for system_id in range(SYSTEM_NAMES.size()):
		var diagnostic: Dictionary = organism.duplicate(true)
		var role_row: Array = diagnostic.get("physiology_role", [])[system_id]
		var core_cells := 0
		for index in range(role_row.size()):
			if int(role_row[index]) == 1:
				diagnostic["alive"][index] = false
				diagnostic["health"][index] = 0.0
				core_cells += 1
		for bond_index in range(diagnostic["bond_ab"].size()):
			var pair: Array = diagnostic["bond_ab"][bond_index]
			if not diagnostic["alive"][int(pair[0])] or not diagnostic["alive"][int(pair[1])]: diagnostic["bond_alive"][bond_index] = false
		var capacity := _compute_physiology_capacities(diagnostic)
		failures[SYSTEM_NAMES[system_id]] = {"core_cells": core_cells, "remaining_capacity": float(capacity.get(SYSTEM_NAMES[system_id], 1.0)), "capacities": capacity}
	return failures


func _prepare_physiology(organism: Dictionary, delta: float) -> void:
	organism["physiology_clock"] = float(organism.get("physiology_clock", 0.0)) - delta
	if float(organism["physiology_clock"]) <= 0.0:
		var capacities := _compute_physiology_capacities(organism)
		if not capacities.is_empty(): organism["physiology_capacities"] = capacities
		organism["physiology_network_reachable"] = _physiology_networks(organism)
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
	var chain_depth: Array = organism.get("motion_chain_depth", [])
	var networks: Array = organism.get("physiology_network_reachable", [])
	var motor_reachable: Dictionary = networks[5] if networks.size() > 5 else {}; var circulation_reachable: Dictionary = networks[0] if networks.size() > 0 else {}
	var squash := float(drivers.get("body_squash", 0.0)); var work := 0.0
	for index in range(organism["position"].size()):
		if not organism["alive"][index] or not reachable.has(str(index)): continue
		var local: Vector2 = rest_local[index]
		local.x *= 1.0 + squash * 0.10; local.y *= 1.0 - squash * 0.08
		local += Vector2(float(drivers.get("body_sway", 0.0)) * 6.5, float(drivers.get("body_bob", 0.0)) * 8.0)
		var organ_key := str(int(organism["organ_id"][index])); var channel := str(channels.get(organ_key, "chassis")); var amount := _channel_driver(channel, drivers)
		var channel_gain := float(channel_integrity.get(channel, 1.0))
		if channel != "chassis": channel_gain *= float(motor_reachable.get(str(index), 0.0)) * sqrt(maxf(0.0, float(circulation_reachable.get(str(index), 0.0))))
		if absf(amount) > 0.00001 and channel != "chassis" and attachments.has(organ_key):
			var attachment: Dictionary = attachments[organ_key]; var root_cell := int(attachment.get("root_cell", -1))
			if root_cell < 0 or not organism["alive"][root_cell] or not reachable.has(str(root_cell)): continue
			var pivot: Vector2 = attachment.get("pivot", Vector2.ZERO)
			var depth := float(chain_depth[index]) if index < chain_depth.size() else 1.0
			var bend_gain := 0.20 + 0.80 * smoothstep(0.0, 1.0, depth)
			local = pivot + (local - pivot).rotated(amount * channel_gain * bend_gain * deg_to_rad(42.0))
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
	var organism_last_frame := int(organism.get("motion_last_event_frame", -1))
	if int(state["frame_index"]) != organism_last_frame:
		organism["motion_last_event_frame"] = int(state["frame_index"])
		for event in state["clip"].get("events", []):
			if int(event.get("frame", -1)) == int(state["frame_index"]):
				organism["motion_event_count"] = int(organism.get("motion_event_count", 0)) + 1
				if not bool(organism.get("motion_autonomous", false)) or organism == organisms[0]: _event("MOTOR EVENT // " + str(event.get("name", "?")).to_upper(), CYAN)


func _break_bond(organism: Dictionary, bond_index: int) -> void:
	var was_alive := bool(organism["bond_alive"][bond_index])
	super(organism, bond_index)
	if was_alive and organism.has("trauma_bond_age") and bond_index < organism["trauma_bond_age"].size():
		organism["trauma_bond_age"][bond_index] = 0.000001
		var pair: Array = organism["bond_ab"][bond_index]
		for cell_index in [int(pair[0]), int(pair[1])]: organism["trauma_wound_age"][cell_index] = maxf(float(organism["trauma_wound_age"][cell_index]), 0.000001)


func _trauma_components(organism: Dictionary) -> Array:
	var components: Array = []; var unseen: Dictionary = {}
	for index in range(organism["alive"].size()):
		if organism["alive"][index]: unseen[str(index)] = true
	while not unseen.is_empty():
		var seed := int(unseen.keys().min()); var queue: Array[int] = [seed]; var component: Array[int] = []; unseen.erase(str(seed)); var cursor := 0
		while cursor < queue.size():
			var current := queue[cursor]; cursor += 1; component.append(current)
			for edge in organism["motion_neighbors"][current]:
				var neighbor := int(edge[0]); var bond_index := int(edge[1])
				if organism["bond_alive"][bond_index] and organism["alive"][neighbor] and unseen.has(str(neighbor)):
					unseen.erase(str(neighbor)); queue.append(neighbor)
		component.sort(); components.append(component)
	components.sort_custom(func(left: Array, right: Array): return left.size() > right.size() if left.size() != right.size() else int(left[0]) < int(right[0]))
	return components


func _component_key(component: Array) -> String:
	var values: PackedStringArray = []
	for index in component: values.append(str(int(index)))
	return ",".join(values)


func _main_component(organism: Dictionary, components: Array) -> Array:
	if components.is_empty(): return []
	var roles: Array = organism.get("physiology_role", []); var circulation: Array = roles[0] if not roles.is_empty() else []
	for component in components:
		for index in component:
			if int(index) < circulation.size() and int(circulation[int(index)]) == 1 and organism["alive"][int(index)]: return component
	return components[0]


func _step_trauma_magnetism(organism: Dictionary, delta: float) -> void:
	var profile: Dictionary = organism.get("trauma_profile", {}); var window := float(profile.get("reconnect_window_seconds", 0.0)); var radius := float(profile.get("magnetic_radius_cells", 0.0)) * CELL_SCALE
	if window <= 0.0 or radius <= 0.0: return
	for bond_index in range(organism["bond_ab"].size()):
		if organism["bond_alive"][bond_index]: continue
		var pair: Array = organism["bond_ab"][bond_index]; var a := int(pair[0]); var b := int(pair[1])
		if not organism["alive"][a] or not organism["alive"][b]: continue
		organism["trauma_bond_age"][bond_index] = float(organism["trauma_bond_age"][bond_index]) + delta
		var age := float(organism["trauma_bond_age"][bond_index])
		if age > window or str(organism["trauma_fragment_fate"][a]) != "attached" or str(organism["trauma_fragment_fate"][b]) != "attached": continue
		var difference: Vector2 = organism["position"][b] - organism["position"][a]; var distance := difference.length()
		if distance > radius or distance <= 0.0001: continue
		var time_gain := 1.0 - age / window; var distance_gain := 1.0 - distance / radius; var weight := float(organism["trauma_bond_magnetic_weight"][bond_index]); var force := difference.normalized() * weight * time_gain * distance_gain * 52.0 * delta
		organism["velocity"][a] += force / maxf(0.1, float(organism["mass"][a])); organism["velocity"][b] -= force / maxf(0.1, float(organism["mass"][b]))
		if distance <= CELL_SCALE * 0.72 and weight > 0.01:
			organism["bond_alive"][bond_index] = true; organism["trauma_bond_age"][bond_index] = 0.0; organism["open_bonds"][a] = maxi(0, int(organism["open_bonds"][a]) - 1); organism["open_bonds"][b] = maxi(0, int(organism["open_bonds"][b]) - 1)
			for cell_index in [a, b]:
				organism["trauma_clot"][cell_index] = maxf(float(organism["trauma_clot"][cell_index]), 0.72)
				organism["trauma_scar"][cell_index] = clampf(float(organism["trauma_scar"][cell_index]) + float(organism["trauma_scar_bias"][cell_index]) * 0.28, 0.0, 1.0)
			organism["trauma_reconnections"] = int(organism["trauma_reconnections"]) + 1


func _step_trauma_components(organism: Dictionary, delta: float) -> void:
	var components := _trauma_components(organism); var main := _main_component(organism, components); var main_key := _component_key(main); var previous: Dictionary = organism.get("trauma_component_age", {}); var current: Dictionary = {}; var profile: Dictionary = organism.get("trauma_profile", {})
	var window := float(profile.get("reconnect_window_seconds", 0.0)); var minimum := int(profile.get("polyp_min_cells", 9999)); var desired := str(profile.get("detached_fate", "biomass"))
	for component in components:
		var key := _component_key(component)
		if key == main_key: continue
		var age := float(previous.get(key, 0.0)) + delta; current[key] = age
		if age < window: continue
		var fate := desired if desired != "biomass" and component.size() >= minimum else "biomass"; var newly_terminal := false
		for index in component:
			if str(organism["trauma_fragment_fate"][int(index)]) == "attached": organism["trauma_fragment_fate"][int(index)] = fate; newly_terminal = true
		if newly_terminal:
			if fate == "biomass": organism["trauma_biomass_components"] = int(organism["trauma_biomass_components"]) + 1
			else: organism["trauma_polyps"] = int(organism["trauma_polyps"]) + 1
		if fate == "biomass":
			for index in component:
				if organism["alive"][int(index)]:
					organism["health"][int(index)] = float(organism["health"][int(index)]) - (0.08 if str(profile.get("family", "")) == "humanoid" else 0.10) * delta
					if float(organism["health"][int(index)]) <= 0.0: organism["health"][int(index)] = 0.0; organism["alive"][int(index)] = false
	organism["trauma_component_age"] = current


func _step_trauma_after(organism: Dictionary, delta: float, previous_health: Array) -> void:
	var capacity: Dictionary = organism.get("physiology_capacities", {}); var circulation := float(capacity.get("circulation", 0.0)); var immune := float(capacity.get("immune", 0.0)); var clot_rate := float(organism.get("trauma_profile", {}).get("clot_rate", 0.0))
	var networks: Array = organism.get("physiology_network_reachable", []); var circulation_delivery: Dictionary = networks[0] if networks.size() > 0 else {}; var immune_delivery: Dictionary = networks[7] if networks.size() > 7 else {}
	for index in range(organism["alive"].size()):
		if not organism["alive"][index]: continue
		var health_ratio := float(organism["health"][index]) / maxf(0.001, float(organism["max_health"][index])); var exposed := int(organism["open_bonds"][index]) > 0 or health_ratio < 0.999
		if exposed:
			organism["trauma_wound_age"][index] = float(organism["trauma_wound_age"][index]) + delta
			var local_circulation := circulation * float(circulation_delivery.get(str(index), 0.0)); var local_immune := immune * float(immune_delivery.get(str(index), 0.0))
			var gain := float(organism["trauma_clotting_weight"][index]) * clot_rate * local_circulation * (0.18 + 0.82 * local_immune) * delta
			organism["trauma_clot"][index] = clampf(float(organism["trauma_clot"][index]) + gain, 0.0, 1.0)
		var healed := maxf(0.0, float(organism["health"][index]) - float(previous_health[index]))
		if healed > 0.0 and exposed: organism["trauma_scar"][index] = clampf(float(organism["trauma_scar"][index]) + float(organism["trauma_scar_bias"][index]) * healed * 0.16, 0.0, 1.0)
	_step_trauma_components(organism, delta)


func _step_organism(organism: Dictionary, delta: float) -> void:
	_prepare_physiology(organism, delta)
	_apply_motion_force(organism, delta)
	_step_trauma_magnetism(organism, delta)
	var previous_health: Array = organism["health"].duplicate()
	super(organism, delta)
	_advance_physiology(organism, delta)
	_step_trauma_after(organism, delta, previous_health)


func _can_reproduce(organism: Dictionary) -> bool:
	return float(organism.get("physiology_capacities", {}).get("reproduction", 0.0)) >= 0.62 and super(organism)


func _draw_organism(organism: Dictionary) -> void:
	if view_mode == 5: _draw_physiology_network(organism)
	super(organism)


func _draw_physiology_network(organism: Dictionary) -> void:
	var roles: Array = organism.get("physiology_role", [])
	if selected_system < 0 or selected_system >= roles.size(): return
	var role_row: Array = roles[selected_system]; var reachable := _physiology_reachable(organism, role_row, selected_system)
	organism["physiology_view_reachable"] = reachable
	var color: Color = SYSTEM_COLORS[selected_system]
	for bond_index in range(organism["bond_ab"].size()):
		var pair: Array = organism["bond_ab"][bond_index]; var a := int(pair[0]); var b := int(pair[1])
		if int(role_row[a]) == 0 or int(role_row[b]) == 0: continue
		var connected: bool = bool(organism["bond_alive"][bond_index]) and bool(organism["alive"][a]) and bool(organism["alive"][b]); var delivery := minf(float(reachable.get(str(a), 0.0)), float(reachable.get(str(b), 0.0)))
		var bond_color := Color(color.r, color.g, color.b, 0.18 + 0.34 * delivery).lerp(Color(1.0, 0.16, 0.28, 0.72), 1.0 - delivery) if connected else Color(1.0, 0.16, 0.28, 0.72)
		draw_line(organism["position"][a], organism["position"][b], bond_color, 1.35)


func _cell_color(organism: Dictionary, index: int) -> Color:
	var color: Color
	if view_mode == 5:
		var roles: Array = organism.get("physiology_role", []); var role := int(roles[selected_system][index]) if selected_system >= 0 and selected_system < roles.size() else 0
		color = Color("#101b24")
		if role > 0:
			color = SYSTEM_COLORS[selected_system]
			if role == 1: color = color.lerp(Color.WHITE, 0.68)
			elif role == 3: color = color.lerp(Color("#fff2a8"), 0.38)
			var delivery := float(organism.get("physiology_view_reachable", {}).get(str(index), 0.0))
			color = color.lerp(Color("#ff274d"), (1.0 - delivery) * 0.72)
	else:
		color = super(organism, index)
	if int(organism["emission"][index]) > 0:
		var pulse: float = clampf(float(organism.get("motion_emission_pulse", 0.0)), 0.0, 1.0)
		color = color.lerp(Color.WHITE, pulse * 0.24)
	var scar := clampf(float(organism.get("trauma_scar", [])[index]), 0.0, 1.0) if organism.get("trauma_scar", []).size() == organism["alive"].size() else 0.0
	if scar > 0.0: color = color.lerp(Color("#6d7782"), scar * 0.72)
	var fate := str(organism.get("trauma_fragment_fate", [])[index]) if organism.get("trauma_fragment_fate", []).size() == organism["alive"].size() else "attached"
	if fate.contains("polyp"): color = color.lerp(Color("#b8ff58"), 0.28)
	if fate == "biomass": color = color.lerp(Color("#6b2e48"), 0.45)
	return color


func _select_motion(delta: int) -> void:
	selected_motion = posmod(selected_motion + delta, EXPECTED_MOTIONS.size()); motion_epoch = simulation_time; last_event_frame = -1
	_event("MOTION // " + str(EXPECTED_MOTIONS[selected_motion]).to_upper(), LIME); _refresh_motion_overlay()


func _select_facing(delta: int) -> void:
	selected_facing = posmod(selected_facing + delta, EXPECTED_FACINGS.size()); motion_epoch = simulation_time; last_event_frame = -1
	_event("FACING // " + str(EXPECTED_FACINGS[selected_facing]).to_upper(), LIME); _refresh_motion_overlay()


func _select_system(delta: int) -> void:
	selected_system = posmod(selected_system + delta, SYSTEM_NAMES.size())
	_event("SYSTEM // " + str(SYSTEM_NAMES[selected_system]).to_upper(), SYSTEM_COLORS[selected_system])
	_refresh_motion_overlay(); queue_redraw()


func _cycle_view() -> void:
	view_mode = (view_mode + 1) % MOTION_VIEW_NAMES.size()
	view_label.text = "VIEW // " + MOTION_VIEW_NAMES[view_mode]
	_refresh_motion_overlay(); queue_redraw()


func _refresh_motion_overlay() -> void:
	if motion_label == null: return
	if _uses_autonomous_motion():
		var states: Dictionary = {}
		for organism in organisms:
			var index := int(organism.get("motion_index", 0)); var name := str(EXPECTED_MOTIONS[index]) if index >= 0 and index < EXPECTED_MOTIONS.size() else "?"
			states[name] = int(states.get(name, 0)) + 1
		var state_labels: Array[String] = []
		for name in states: state_labels.append("%s:%d" % [str(name).to_upper(), int(states[name])])
		state_labels.sort()
		motion_label.text = "AUTONOMOUS // " + "  ".join(state_labels)
		if organisms.is_empty(): driver_label.text = "PER-ORGANISM MOTION // ORGAN CAPACITY GATED"; return
		var first_capacity: Dictionary = organisms[0].get("physiology_capacities", {})
		driver_label.text = "%s // BRAIN %3d  HEART %3d  LUNG %3d  LIMB %3d" % [str(organisms[0].get("motion_behavior", "?")).to_upper(), int(float(first_capacity.get("neural", 0.0)) * 100.0), int(float(first_capacity.get("circulation", 0.0)) * 100.0), int(float(first_capacity.get("respiration", 0.0)) * 100.0), int(float(first_capacity.get("locomotion", 0.0)) * 100.0)]
		if view_mode == 5: driver_label.text += " // %s %d" % [str(SYSTEM_NAMES[selected_system]).to_upper(), int(float(first_capacity.get(SYSTEM_NAMES[selected_system], 0.0)) * 100.0)]
		return
	var clip := _current_clip(int(organisms[0]["data"].get("family_id", 0))) if not organisms.is_empty() else {}
	motion_label.text = "%s // %s // %s" % [str(EXPECTED_MOTIONS[selected_motion]).to_upper(), str(EXPECTED_FACINGS[selected_facing]).to_upper(), "%d FPS" % int(clip.get("fps", 0))]
	if organisms.is_empty(): driver_label.text = "W/S MOTION  ARROWS FACING  //  ORGAN TARGETS + LIVE SPRINGS"; return
	var capacity: Dictionary = organisms[0].get("physiology_capacities", {})
	var scar_mean := _sum_float(organisms[0].get("trauma_scar", [])) / maxf(1.0, float(organisms[0]["alive"].size()))
	driver_label.text = "BRAIN %3d  HEART %3d  LUNG %3d  GUT %3d  SCAR %2d  POLYP %d" % [int(float(capacity.get("neural", 0.0)) * 100.0), int(float(capacity.get("circulation", 0.0)) * 100.0), int(float(capacity.get("respiration", 0.0)) * 100.0), int(float(capacity.get("digestion", 0.0)) * 100.0), int(scar_mean * 100.0), int(organisms[0].get("trauma_polyps", 0))]
	if view_mode == 5: driver_label.text += " // %s %d" % [str(SYSTEM_NAMES[selected_system]).to_upper(), int(float(capacity.get(SYSTEM_NAMES[selected_system], 0.0)) * 100.0)]


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
			KEY_C: _select_system(1)


func _diagnostic_detach_component(organism: Dictionary, minimum_cells: int) -> Array:
	var groups: Dictionary = {}
	for index in range(organism["organ_id"].size()):
		var key := str(int(organism["organ_id"][index])); if not groups.has(key): groups[key] = []
		groups[key].append(index)
	var candidates: Array = groups.values(); candidates.sort_custom(func(left: Array, right: Array): return left.size() < right.size())
	for component in candidates:
		if component.size() < minimum_cells or component.size() >= organism["alive"].size() / 2: continue
		var membership: Dictionary = {}; for index in component: membership[str(int(index))] = true
		var boundary: Array[int] = []
		for bond_index in range(organism["bond_ab"].size()):
			var pair: Array = organism["bond_ab"][bond_index]
			if membership.has(str(int(pair[0]))) != membership.has(str(int(pair[1]))): boundary.append(bond_index)
		if boundary.is_empty(): continue
		for bond_index in boundary: _break_bond(organism, bond_index)
		for observed in _trauma_components(organism):
			if observed.size() == component.size() and _component_key(observed) == _component_key(component): return observed
	return []


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
	var actuation_velocity := 0.0; var trauma := {"killed": 0, "bonds": 0}; var population := organisms.size(); var physiology_core_damage_verified := false; var all_system_core_failures_verified := false; var system_core_failures: Dictionary = {}; var system_view_verified := false; var member_routing_verified := false; var graded_local_delivery_verified := false; var local_perfusion_verified := false; var progressive_chain_verified := false; var trauma_reconnection_verified := false; var plant_polyp_verified := false
	if not organisms.is_empty():
		var baseline_capacity := _compute_physiology_capacities(organisms[0])
		if baseline_capacity.size() != 8 or baseline_capacity.values().any(func(value): return float(value) < 0.99): errors.append("physiology baseline")
		var baseline_networks := _physiology_networks(organisms[0]); member_routing_verified = baseline_networks.size() == SYSTEM_NAMES.size()
		for system_id in range(mini(baseline_networks.size(), SYSTEM_NAMES.size())):
			var role_row: Array = organisms[0].get("physiology_role", [])[system_id]
			for key in baseline_networks[system_id]:
				var delivery := float(baseline_networks[system_id][key])
				if int(role_row[int(key)]) <= 0 or delivery <= 0.0 or delivery > 1.0: member_routing_verified = false
		if not member_routing_verified: errors.append("member-restricted routing")
		var graded_diagnostic: Dictionary = organisms[0].duplicate(true); var motor_roles: Array = graded_diagnostic.get("physiology_role", [])[5]; var graded_cell := -1
		for index in range(motor_roles.size()):
			if int(motor_roles[index]) == 3: graded_cell = index; break
		if graded_cell >= 0:
			graded_diagnostic["health"][graded_cell] = float(graded_diagnostic["max_health"][graded_cell]) * 0.37
			var graded_network := _physiology_reachable(graded_diagnostic, motor_roles, 5); var graded_capacity := _compute_physiology_capacities(graded_diagnostic)
			graded_local_delivery_verified = absf(float(graded_network.get(str(graded_cell), 0.0)) - 0.37) < 0.0001 and float(graded_capacity.get("locomotion", 1.0)) < float(baseline_capacity.get("locomotion", 0.0)) and float(graded_capacity.get("locomotion", 0.0)) > 0.0
		if not graded_local_delivery_verified: errors.append("graded local physiology")
		var perfusion_diagnostic: Dictionary = organisms[0].duplicate(true); var circulation_roles: Array = perfusion_diagnostic.get("physiology_role", [])[0]; var perfusion_cell := -1
		for index in range(circulation_roles.size()):
			if int(circulation_roles[index]) == 2 and float(perfusion_diagnostic.get("fluid_baseline", [])[index]) > 0.0: perfusion_cell = index; break
		if perfusion_cell >= 0:
			perfusion_diagnostic["fluid"][perfusion_cell] = float(perfusion_diagnostic["fluid_baseline"][perfusion_cell]) * 0.22
			var perfusion_network := _physiology_reachable(perfusion_diagnostic, circulation_roles, 0); var perfusion_capacity := _compute_physiology_capacities(perfusion_diagnostic)
			local_perfusion_verified = absf(float(perfusion_network.get(str(perfusion_cell), 0.0)) - 0.22) < 0.0001 and float(perfusion_capacity.get("circulation", 1.0)) < float(baseline_capacity.get("circulation", 0.0)) and float(perfusion_capacity.get("circulation", 0.0)) > 0.0
		if not local_perfusion_verified: errors.append("local fluid perfusion")
		var depths: Array = organisms[0].get("motion_chain_depth", [])
		progressive_chain_verified = depths.any(func(value): return float(value) > 0.15 and float(value) < 0.85) and depths.any(func(value): return float(value) > 0.90)
		if not progressive_chain_verified: errors.append("progressive appendage chains")
		var original_view := view_mode; var original_system := selected_system; var view_cells := 0
		view_mode = 5
		for system_id in range(SYSTEM_NAMES.size()):
			selected_system = system_id; var role_row: Array = organisms[0].get("physiology_role", [])[system_id]
			organisms[0]["physiology_view_reachable"] = _physiology_reachable(organisms[0], role_row, system_id)
			for index in range(role_row.size()):
				if int(role_row[index]) > 0:
					var view_color := _cell_color(organisms[0], index)
					if not is_finite(view_color.r + view_color.g + view_color.b): errors.append("system view color")
					view_cells += 1
		view_mode = original_view; selected_system = original_system
		system_view_verified = view_cells > 0
		if not system_view_verified: errors.append("system view membership")
		system_core_failures = _diagnostic_system_core_failures(organisms[0])
		all_system_core_failures_verified = system_core_failures.size() == SYSTEM_NAMES.size()
		for system_name in SYSTEM_NAMES:
			var failure: Dictionary = system_core_failures.get(system_name, {})
			if int(failure.get("core_cells", 0)) <= 0 or float(failure.get("remaining_capacity", 1.0)) > 0.000001: all_system_core_failures_verified = false
		if not all_system_core_failures_verified: errors.append("physiology system core failures")
		selected_motion = EXPECTED_MOTIONS.find("locomote"); selected_facing = 2; motion_epoch = 0.0; simulation_time = 0.25
		_apply_motion_force(organisms[0], 1.0 / 60.0)
		for velocity in organisms[0]["velocity"]: actuation_velocity += velocity.length()
		if not is_finite(actuation_velocity) or actuation_velocity <= 0.01: errors.append("neuromuscular actuation")
		var reconnect_bond := 0; var reconnect_pair: Array = organisms[0]["bond_ab"][reconnect_bond]; var reconnect_a := int(reconnect_pair[0]); var reconnect_b := int(reconnect_pair[1]); _break_bond(organisms[0], reconnect_bond)
		organisms[0]["position"][reconnect_b] = organisms[0]["position"][reconnect_a] + Vector2(CELL_SCALE * 0.5, 0.0); _step_trauma_magnetism(organisms[0], 1.0 / 60.0)
		trauma_reconnection_verified = organisms[0]["bond_alive"][reconnect_bond] and int(organisms[0]["trauma_reconnections"]) == 1 and maxf(float(organisms[0]["trauma_scar"][reconnect_a]), float(organisms[0]["trauma_scar"][reconnect_b])) > 0.0
		if not trauma_reconnection_verified: errors.append("trauma reconnection")
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
	var plant_index := -1
	for index in range(catalog.get("species", []).size()):
		if int(catalog.get("species", [])[index].get("family_id", -1)) == 2: plant_index = index; break
	if plant_index >= 0:
		var previous_species := selected_species; selected_species = plant_index; var plant_data := _load_species_data(plant_index); var plant := _create_organism(plant_data, Vector2(1000, 360), 0, 0); selected_species = previous_species
		var minimum := int(plant.get("trauma_profile", {}).get("polyp_min_cells", 4)); var detached := _diagnostic_detach_component(plant, minimum)
		if not detached.is_empty():
			_step_trauma_components(plant, float(plant["trauma_profile"].get("reconnect_window_seconds", 15.0)) + 0.1)
			plant_polyp_verified = detached.all(func(index): return str(plant["trauma_fragment_fate"][int(index)]) == "polyp")
	if not plant_polyp_verified: errors.append("plant polyp fate")
	var report := {
		"format": "nullvector-cellular-motion-godot-smoke-v7", "passed": errors.is_empty(), "errors": errors,
		"engine": Engine.get_version_info().get("string", ""), "motion_bundle_id": motion_catalog.get("bundle_id", ""),
		"physiology_bundle_id": physiology_catalog.get("bundle_id", ""), "trauma_bundle_id": trauma_catalog.get("bundle_id", ""),
		"organism_bundle_id": catalog.get("bundle_id", ""), "identity_count": motion_catalog.get("identity_count", 0),
		"physiology_identity_count": physiology_catalog.get("identity_count", 0), "physiology_system_count": physiology_catalog.get("system_count", 0),
		"trauma_identity_count": trauma_catalog.get("identity_count", 0), "physiology_core_damage_verified": physiology_core_damage_verified,
		"all_system_core_failures_verified": all_system_core_failures_verified, "system_view_verified": system_view_verified,
		"member_routing_verified": member_routing_verified, "graded_local_delivery_verified": graded_local_delivery_verified, "local_perfusion_verified": local_perfusion_verified, "progressive_chain_verified": progressive_chain_verified,
		"system_core_failures": system_core_failures, "trauma_reconnection_verified": trauma_reconnection_verified,
		"plant_polyp_verified": plant_polyp_verified, "clip_count": clip_count, "frame_count": frame_count,
		"event_count": event_count, "mapped_organs": mapped_organs, "actuation_velocity": actuation_velocity,
		"damage_killed": trauma["killed"], "damage_bonds": trauma["bonds"], "population_after_reproduction": population,
		"python_runtime_required": false,
	}
	if not report_path.is_empty():
		var file := FileAccess.open(report_path, FileAccess.WRITE)
		if file != null: file.store_string(JSON.stringify(report, "  ", true) + "\n")
	if errors.is_empty(): print("CELLULAR_MOTION_SMOKE_OK identities=45 clips=%d frames=%d organs=%d velocity=%.3f population=%d" % [clip_count, frame_count, mapped_organs, actuation_velocity, population])
	else: push_error("CELLULAR_MOTION_SMOKE_FAIL " + ", ".join(errors))
	get_tree().quit(0 if errors.is_empty() else 1)
