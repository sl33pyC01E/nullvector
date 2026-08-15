class_name CreatureStageInterventionCorpus
extends RefCounted

const Neural = preload("res://scripts/creature_stage/creature_neural.gd")
const Creature = preload("res://scripts/creature_stage/neural_creature.gd")

const FORMAT := "nullvector-creature-stage-intervention-corpus-v1"
const FIXED_HZ := 30
const FRAMES_PER_CLIP := 180
const DELTA := 1.0 / float(FIXED_HZ)
const INTERVENTION_FRAME := 15
const HEAL_FRAME := 75
const POSITION_SCALE := 256
const POSITION_BIAS := 32768
const UNIT_SCALE := 65535
const FLUID_SCALAR_SCALE := 1024
const MAX_FLUIDS := 160
const MINIMUM_FREE_BYTES := 100 * 1024 * 1024 * 1024
const PUBLICATION_RESERVE_BYTES := 128 * 1024 * 1024
const SUMMARY_FIELDS := [
	"integrity", "neural", "circulation", "respiration", "digestion",
	"senses", "energy", "hydration", "dead", "fluid_count",
]
const INTERVENTIONS := [
	{"name": "control", "target": "none", "event_frames": []},
	{"name": "wound", "target": "central_nonorgan_patch", "event_frames": [INTERVENTION_FRAME]},
	{"name": "heal", "target": "central_nonorgan_patch", "event_frames": [INTERVENTION_FRAME, HEAL_FRAME]},
	{"name": "cut", "target": "lower_appendage_plane", "event_frames": [INTERVENTION_FRAME]},
	{"name": "neural_ablation", "target": "neural", "event_frames": [INTERVENTION_FRAME]},
	{"name": "circulation_ablation", "target": "circulation", "event_frames": [INTERVENTION_FRAME]},
	{"name": "respiration_ablation", "target": "respiration", "event_frames": [INTERVENTION_FRAME]},
	{"name": "digestion_ablation", "target": "digestion", "event_frames": [INTERVENTION_FRAME]},
	{"name": "sensory_ablation", "target": "senses", "event_frames": [INTERVENTION_FRAME]},
]
const ORGAN_GROUPS := {
	"neural": ["brain", "meristem", "phase_brain", "processor"],
	"circulation": ["heart", "vascular", "flux", "coolant_pump"],
	"respiration": ["lung", "frond", "orbital", "radiator"],
	"digestion": ["gut", "bulb", "transmuter", "battery"],
	"senses": ["eye", "photoreceptor", "singularity", "optic"],
}


static func _sha256_bytes(bytes: PackedByteArray) -> String:
	var context := HashingContext.new()
	if context.start(HashingContext.HASH_SHA256) != OK:
		return "hash-error"
	context.update(bytes)
	return context.finish().hex_encode()


static func _sha256_text(value: String) -> String:
	return _sha256_bytes(value.to_utf8_buffer())


static func _sha256_file(path: String) -> String:
	if not FileAccess.file_exists(path):
		return "missing"
	return _sha256_bytes(FileAccess.get_file_as_bytes(path))


static func _append_u16(buffer: PackedByteArray, value: int) -> void:
	buffer.append(value & 0xff)
	buffer.append((value >> 8) & 0xff)


static func _encode_unit(value: float) -> int:
	return clampi(roundi(clampf(value, 0.0, 1.0) * UNIT_SCALE), 0, UNIT_SCALE)


static func _encode_signed(value: float, scale: int, counters: Dictionary) -> int:
	var raw := roundi(value * float(scale)) + POSITION_BIAS
	var encoded := clampi(raw, 0, 65535)
	if encoded != raw:
		counters["clipped_values"] = int(counters["clipped_values"]) + 1
	return encoded


static func _encode_scalar(value: float, counters: Dictionary) -> int:
	var raw := roundi(maxf(value, 0.0) * float(FLUID_SCALAR_SCALE))
	var encoded := clampi(raw, 0, 65535)
	if encoded != raw:
		counters["clipped_values"] = int(counters["clipped_values"]) + 1
	return encoded


static func _cell_metadata(blueprint: Dictionary) -> Dictionary:
	var cells: Array = []
	var identity_material := PackedStringArray()
	for source in blueprint["cells"]:
		var grid: Vector2i = source["grid"]
		var record := {
			"grid": [grid.x, grid.y],
			"tissue": str(source.get("tissue", "skin")),
			"organ": str(source.get("organ", "none")),
			"appendage": int(source.get("appendage", -1)),
			"side": int(source.get("side", 0)),
		}
		cells.append(record)
		identity_material.append("%d,%d,%s,%s,%d,%d" % [grid.x, grid.y, record["tissue"], record["organ"], record["appendage"], record["side"]])
	return {"cells": cells, "cell_identity_sha256": _sha256_text("|".join(identity_material))}


static func _source_sha256() -> Dictionary:
	var source_paths := [
		"scripts/creature_stage/creature_neural.gd",
		"scripts/creature_stage/neural_creature.gd",
		"scripts/creature_stage/intervention_corpus.gd",
	]
	var hashes: Dictionary = {}
	var material := PackedStringArray()
	for relative_path in source_paths:
		var digest := _sha256_file(ProjectSettings.globalize_path("res://" + relative_path))
		hashes[relative_path] = digest
		material.append(relative_path + ":" + digest)
	return {"files": hashes, "combined_sha256": _sha256_text("|".join(material))}


static func _central_nonorgan_cell(body: Node2D) -> Dictionary:
	var best: Dictionary = {}
	var best_distance := INF
	for cell in body.cells:
		var organ: String = str(cell.get("organ", "none"))
		var vital := false
		for group in ORGAN_GROUPS.values():
			if organ in group:
				vital = true
				break
		if vital:
			continue
		var distance: float = Vector2(cell["rest"]).length_squared()
		if distance < best_distance:
			best = cell
			best_distance = distance
	return best


static func _ablate_group(body: Node2D, group: String) -> int:
	var positions: Array[Vector2] = []
	for cell in body.cells:
		if str(cell.get("organ", "none")) in ORGAN_GROUPS[group]:
			positions.append(Vector2(cell["pos"]))
	var hits := 0
	for position in positions:
		hits += body.damage_at(position, 0.35, 1.25, Vector2(6.0, -2.0))
	return hits


static func _apply_intervention(body: Node2D, intervention: String, frame: int) -> int:
	if intervention == "heal" and frame == HEAL_FRAME:
		body.heal(0.58)
		return 0
	if frame != INTERVENTION_FRAME:
		return 0
	match intervention:
		"control":
			return 0
		"wound", "heal":
			var target := _central_nonorgan_cell(body)
			if target.is_empty():
				return 0
			return body.damage_at(Vector2(target["pos"]), 4.3, 0.45, Vector2(9.0, -2.0))
		"cut":
			var cut_y: float = body.body_radius * 0.30
			return body.cut_segment(Vector2(-body.body_radius, cut_y), Vector2(body.body_radius, cut_y), 2.25)
		"neural_ablation":
			return _ablate_group(body, "neural")
		"circulation_ablation":
			return _ablate_group(body, "circulation")
		"respiration_ablation":
			return _ablate_group(body, "respiration")
		"digestion_ablation":
			return _ablate_group(body, "digestion")
		"sensory_ablation":
			return _ablate_group(body, "senses")
	return 0


static func _append_frame(buffer: PackedByteArray, body: Node2D, counters: Dictionary) -> void:
	for value in [
		body.alive_fraction(), body.neural_capacity(), body.circulation_capacity(),
		body.respiration_capacity(), body.digestion_capacity(), body.sensory_capacity(),
		body.energy, body.hydration,
	]:
		_append_u16(buffer, _encode_unit(float(value)))
	_append_u16(buffer, UNIT_SCALE if body.dead else 0)
	_append_u16(buffer, mini(body.fluids.size(), MAX_FLUIDS))
	for cell in body.cells:
		var delta_position: Vector2 = Vector2(cell["pos"]) - Vector2(cell["rest"])
		_append_u16(buffer, _encode_signed(delta_position.x, POSITION_SCALE, counters))
		_append_u16(buffer, _encode_signed(delta_position.y, POSITION_SCALE, counters))
		_append_u16(buffer, _encode_unit(float(cell.get("health", 0.0))))
		_append_u16(buffer, UNIT_SCALE if bool(cell.get("alive", true)) else 0)
	for fluid_index in range(MAX_FLUIDS):
		if fluid_index < body.fluids.size():
			var drop: Dictionary = body.fluids[fluid_index]
			var position: Vector2 = drop.get("pos", Vector2.ZERO)
			var velocity: Vector2 = drop.get("velocity", Vector2.ZERO)
			_append_u16(buffer, _encode_signed(position.x, POSITION_SCALE, counters))
			_append_u16(buffer, _encode_signed(position.y, POSITION_SCALE, counters))
			_append_u16(buffer, _encode_signed(velocity.x, POSITION_SCALE, counters))
			_append_u16(buffer, _encode_signed(velocity.y, POSITION_SCALE, counters))
			_append_u16(buffer, _encode_scalar(float(drop.get("radius", 0.0)), counters))
			_append_u16(buffer, _encode_scalar(float(drop.get("life", 0.0)), counters))
		else:
			for _component in range(6):
				_append_u16(buffer, 0)


static func export_corpus(output_path: String) -> Dictionary:
	var destination := output_path if output_path.is_absolute_path() else ProjectSettings.globalize_path(output_path)
	var parent := DirAccess.open(destination.get_base_dir())
	if parent == null:
		return {"passed": false, "error": "destination parent is unavailable for disk guard"}
	if parent.get_space_left() - PUBLICATION_RESERVE_BYTES < MINIMUM_FREE_BYTES:
		return {"passed": false, "error": "100 GiB free-disk floor would be violated"}
	if DirAccess.dir_exists_absolute(destination) or FileAccess.file_exists(destination):
		return {"passed": false, "error": "destination already exists"}
	var staging := destination + ".tmp"
	if DirAccess.dir_exists_absolute(staging) or FileAccess.file_exists(staging):
		return {"passed": false, "error": "staging destination already exists"}
	var mkdir_error := DirAccess.make_dir_recursive_absolute(staging)
	if mkdir_error != OK:
		return {"passed": false, "error": "unable to create staging directory", "code": mkdir_error}

	var binary_path := staging.path_join("intervention_frames.u16le")
	var binary := FileAccess.open(binary_path, FileAccess.WRITE)
	if binary == null:
		return {"passed": false, "error": "unable to open intervention binary"}
	var chassis: Array = []
	var clips: Array = []
	var corpus_identity := PackedStringArray()
	var counters := {"clipped_values": 0, "maximum_fluids": 0}
	var byte_offset := 0
	for family_id in range(5):
		for morphotype_id in range(4):
			var chassis_id := family_id * 4 + morphotype_id
			var specimen_seed := 0x710D0000 + family_id * 0x100 + morphotype_id
			var blueprint := Neural.decode_morphology(family_id, specimen_seed, 0)
			var metadata := _cell_metadata(blueprint)
			chassis.append({
				"chassis_id": chassis_id,
				"family": Neural.FAMILIES[family_id],
				"family_id": family_id,
				"morphotype": blueprint["morphotype"],
				"morphotype_id": morphotype_id,
				"seed": specimen_seed,
				"generation": 0,
				"genes": blueprint["genes"],
				"cell_count": metadata["cells"].size(),
				"cell_identity_sha256": metadata["cell_identity_sha256"],
				"cells": metadata["cells"],
			})
			corpus_identity.append("chassis:%d:%s" % [chassis_id, metadata["cell_identity_sha256"]])
			for intervention_id in range(INTERVENTIONS.size()):
				var intervention: Dictionary = INTERVENTIONS[intervention_id]
				seed(0x41C00000 + chassis_id * 0x100 + intervention_id)
				var body: Node2D = Creature.new()
				body.configure(blueprint)
				body.set_commands(Vector2.ZERO, Vector2(0.8, -0.6).normalized(), 0.0, 0.0, 0.0)
				var clip_bytes := PackedByteArray()
				var hit_count := 0
				var maximum_fluids := 0
				for frame in range(FRAMES_PER_CLIP):
					hit_count += _apply_intervention(body, str(intervention["name"]), frame)
					maximum_fluids = maxi(maximum_fluids, body.fluids.size())
					_append_frame(clip_bytes, body, counters)
					body.simulate_body(DELTA)
				counters["maximum_fluids"] = maxi(int(counters["maximum_fluids"]), maximum_fluids)
				var frame_stride: int = 20 + body.cells.size() * 8 + MAX_FLUIDS * 12
				var trajectory_sha256 := _sha256_bytes(clip_bytes)
				binary.store_buffer(clip_bytes)
				var record := {
					"clip_id": clips.size(),
					"chassis_id": chassis_id,
					"family_id": family_id,
					"morphotype_id": morphotype_id,
					"intervention": intervention["name"],
					"intervention_id": intervention_id,
					"target": intervention["target"],
					"event_frames": intervention["event_frames"],
					"hit_count": hit_count,
					"maximum_fluid_count": maximum_fluids,
					"frames": FRAMES_PER_CLIP,
					"cell_count": body.cells.size(),
					"frame_stride_bytes": frame_stride,
					"byte_offset": byte_offset,
					"byte_length": clip_bytes.size(),
					"trajectory_sha256": trajectory_sha256,
				}
				clips.append(record)
				corpus_identity.append("clip:%d:%s" % [record["clip_id"], trajectory_sha256])
				byte_offset += clip_bytes.size()
				body.free()
	binary.close()

	if clips.size() != 180 or int(counters["clipped_values"]) != 0 or int(counters["maximum_fluids"]) > MAX_FLUIDS:
		return {"passed": false, "error": "coverage, quantization, or fluid bound failed", "clips": clips.size(), "counters": counters}
	var binary_sha256 := _sha256_file(binary_path)
	var source := _source_sha256()
	var manifest := {
		"format": FORMAT,
		"passed": true,
		"fixed_hz": FIXED_HZ,
		"frames_per_clip": FRAMES_PER_CLIP,
		"intervention_frame": INTERVENTION_FRAME,
		"heal_frame": HEAL_FRAME,
		"family_count": 5,
		"chassis_count": chassis.size(),
		"intervention_count": INTERVENTIONS.size(),
		"clip_count": clips.size(),
		"total_frames": clips.size() * FRAMES_PER_CLIP,
		"total_cell_samples": clips.reduce(func(total: int, clip: Dictionary) -> int: return total + int(clip["cell_count"]) * FRAMES_PER_CLIP, 0),
		"maximum_fluid_slots": MAX_FLUIDS,
		"summary_fields": SUMMARY_FIELDS,
		"interventions": INTERVENTIONS,
		"organ_groups": ORGAN_GROUPS,
		"encodings": {
			"position_velocity": {"scale": POSITION_SCALE, "bias": POSITION_BIAS},
			"unit": {"scale": UNIT_SCALE},
			"fluid_scalar": {"scale": FLUID_SCALAR_SCALE},
			"clipped_values": counters["clipped_values"],
		},
		"contracts": {
			"morphology": "coordinate-conditioned-safe-scaffold-v1",
			"physiology": "cellular-organ-causal-scaffold-v1",
			"orientation": "vertical-locked-2.5d-v1",
			"binary": "clip-frame-summary-cell-fluid-u16le-v1",
			"fluid": "ground-plane-diffuse-puddle-v1",
		},
		"source": source,
		"corpus_identity_sha256": _sha256_text("|".join(corpus_identity)),
		"artifacts": {"intervention_frames": {"path": "intervention_frames.u16le", "bytes": byte_offset, "sha256": binary_sha256}},
		"chassis": chassis,
		"clips": clips,
	}
	var manifest_file := FileAccess.open(staging.path_join("manifest.json"), FileAccess.WRITE)
	if manifest_file == null:
		return {"passed": false, "error": "unable to open manifest"}
	manifest_file.store_string(JSON.stringify(manifest, "  ", false))
	manifest_file.close()
	var rename_error := DirAccess.rename_absolute(staging, destination)
	if rename_error != OK:
		return {"passed": false, "error": "atomic publish failed", "code": rename_error, "staging": staging}
	return {
		"passed": true,
		"destination": destination,
		"clip_count": clips.size(),
		"total_frames": manifest["total_frames"],
		"binary_bytes": byte_offset,
		"binary_sha256": binary_sha256,
		"corpus_identity_sha256": manifest["corpus_identity_sha256"],
	}
