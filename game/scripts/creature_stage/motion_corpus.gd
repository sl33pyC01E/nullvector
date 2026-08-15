class_name CreatureStageMotionCorpus
extends RefCounted

const Neural = preload("res://scripts/creature_stage/creature_neural.gd")
const Creature = preload("res://scripts/creature_stage/neural_creature.gd")

const FORMAT := "nullvector-creature-stage-motion-corpus-v1"
const FIXED_HZ := 30
const FRAMES_PER_CLIP := 72
const DELTA := 1.0 / float(FIXED_HZ)
const POSITION_SCALE := 256
const POSITION_BIAS := 32768
const EXPECTED_CLIPS := 260
const MINIMUM_FREE_BYTES := 100 * 1024 * 1024 * 1024
const PUBLICATION_RESERVE_BYTES := 64 * 1024 * 1024


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


static func _append_u16_le(buffer: PackedByteArray, value: int) -> void:
	buffer.append(value & 0xff)
	buffer.append((value >> 8) & 0xff)


static func _configure_body(body: Node2D, motion: String) -> Dictionary:
	var move := Vector2.ZERO
	var aim := Vector2(0.8, -0.6).normalized()
	if motion == "locomote":
		move = Vector2(0.72, 0.34).normalized()
	body.set_commands(move, aim, 0.0, 1.0 if motion == "attack" else 0.0, 0.0)
	if motion == "death":
		body.dead = true
	body.play_motion(motion, 999.0)
	if motion == "attack":
		body.action_impulse = 1.0
	return {
		"move": [move.x, move.y],
		"aim": [aim.x, aim.y],
		"attack": 1.0 if motion == "attack" else 0.0,
		"utility": 1.0 if motion == "cast" else 0.0,
		"external_event": "impact" if motion == "hit" else ("terminal" if motion == "death" else "none"),
	}


static func _cell_metadata(blueprint: Dictionary) -> Dictionary:
	var cells: Array = []
	var identity_material := PackedStringArray()
	for source in blueprint["cells"]:
		var grid: Vector2i = source["grid"]
		var health_q: int = clampi(roundi(float(source.get("health", 1.0)) * 255.0), 0, 255)
		var record := {
			"grid": [grid.x, grid.y],
			"tissue": str(source.get("tissue", "skin")),
			"organ": str(source.get("organ", "none")),
			"appendage": int(source.get("appendage", -1)),
			"side": int(source.get("side", 0)),
			"initial_health_q": health_q,
		}
		cells.append(record)
		identity_material.append("%d,%d,%s,%s,%d,%d,%d" % [grid.x, grid.y, record["tissue"], record["organ"], record["appendage"], record["side"], health_q])
	return {
		"cells": cells,
		"cell_identity_sha256": _sha256_text("|".join(identity_material)),
	}


static func _source_sha256() -> Dictionary:
	var source_paths := [
		"scripts/creature_stage/creature_neural.gd",
		"scripts/creature_stage/neural_creature.gd",
		"scripts/creature_stage/motion_corpus.gd",
	]
	var hashes: Dictionary = {}
	var identity := PackedStringArray()
	for relative_path in source_paths:
		var absolute_path := ProjectSettings.globalize_path("res://" + relative_path)
		var digest := _sha256_file(absolute_path)
		hashes[relative_path] = digest
		identity.append(relative_path + ":" + digest)
	return {"files": hashes, "combined_sha256": _sha256_text("|".join(identity))}


static func export_corpus(output_path: String) -> Dictionary:
	var destination := output_path if output_path.is_absolute_path() else ProjectSettings.globalize_path(output_path)
	var guard_directory := DirAccess.open(destination.get_base_dir())
	if guard_directory == null:
		return {"passed": false, "error": "destination parent is unavailable for disk guard"}
	if guard_directory.get_space_left() - PUBLICATION_RESERVE_BYTES < MINIMUM_FREE_BYTES:
		return {"passed": false, "error": "100 GiB free-disk floor would be violated"}
	if DirAccess.dir_exists_absolute(destination) or FileAccess.file_exists(destination):
		return {"passed": false, "error": "destination already exists", "destination": destination}
	var staging := destination + ".tmp"
	if DirAccess.dir_exists_absolute(staging) or FileAccess.file_exists(staging):
		return {"passed": false, "error": "staging destination already exists", "destination": staging}
	var mkdir_error := DirAccess.make_dir_recursive_absolute(staging)
	if mkdir_error != OK:
		return {"passed": false, "error": "unable to create staging directory", "code": mkdir_error}

	var binary_path := staging.path_join("motion_frames.u16le")
	var binary := FileAccess.open(binary_path, FileAccess.WRITE)
	if binary == null:
		return {"passed": false, "error": "unable to open motion binary"}

	var chassis: Array = []
	var clips: Array = []
	var corpus_identity := PackedStringArray()
	var byte_offset := 0
	var clipped_values := 0
	var minimum_delta_q := 65535
	var maximum_delta_q := 0
	for family_id in range(5):
		for morphotype_id in range(4):
			var chassis_id: int = family_id * 4 + morphotype_id
			var seed := 0x6D0F0000 + family_id * 0x100 + morphotype_id
			var blueprint := Neural.decode_morphology(family_id, seed, 0)
			var metadata := _cell_metadata(blueprint)
			var chassis_record := {
				"chassis_id": chassis_id,
				"family": Neural.FAMILIES[family_id],
				"family_id": family_id,
				"morphotype": blueprint["morphotype"],
				"morphotype_id": morphotype_id,
				"seed": seed,
				"generation": 0,
				"genes": blueprint["genes"],
				"cell_count": metadata["cells"].size(),
				"cell_identity_sha256": metadata["cell_identity_sha256"],
				"cells": metadata["cells"],
			}
			chassis.append(chassis_record)
			corpus_identity.append("chassis:%d:%s" % [chassis_id, metadata["cell_identity_sha256"]])
			for motion_id in range(Creature.MOTIONS.size()):
				var motion: String = Creature.MOTIONS[motion_id]
				var body: Node2D = Creature.new()
				body.configure(blueprint)
				var controls: Dictionary = _configure_body(body, motion)
				var clip_bytes := PackedByteArray()
				clip_bytes.resize(0)
				for _frame in range(FRAMES_PER_CLIP):
					body.simulate_body(DELTA)
					for cell in body.cells:
						var delta_position: Vector2 = Vector2(cell["pos"]) - Vector2(cell["rest"])
						for component in [delta_position.x, delta_position.y]:
							var unbounded: int = roundi(float(component) * POSITION_SCALE) + POSITION_BIAS
							var encoded: int = clampi(unbounded, 0, 65535)
							if encoded != unbounded:
								clipped_values += 1
							minimum_delta_q = mini(minimum_delta_q, encoded)
							maximum_delta_q = maxi(maximum_delta_q, encoded)
							_append_u16_le(clip_bytes, encoded)
				var trajectory_sha256 := _sha256_bytes(clip_bytes)
				binary.store_buffer(clip_bytes)
				var clip_record := {
					"clip_id": clips.size(),
					"chassis_id": chassis_id,
					"family_id": family_id,
					"morphotype_id": morphotype_id,
					"motion": motion,
					"motion_id": motion_id,
					"frames": FRAMES_PER_CLIP,
					"cell_count": body.cells.size(),
					"frame_stride_bytes": body.cells.size() * 4,
					"byte_offset": byte_offset,
					"byte_length": clip_bytes.size(),
					"trajectory_sha256": trajectory_sha256,
					"controls": controls,
				}
				clips.append(clip_record)
				corpus_identity.append("clip:%d:%s" % [clip_record["clip_id"], trajectory_sha256])
				byte_offset += clip_bytes.size()
				body.free()
	binary.close()

	if clips.size() != EXPECTED_CLIPS or clipped_values != 0:
		return {"passed": false, "error": "corpus coverage or quantization bound failed", "clips": clips.size(), "clipped_values": clipped_values}
	var binary_sha256 := _sha256_file(binary_path)
	var source := _source_sha256()
	var manifest := {
		"format": FORMAT,
		"passed": true,
		"fixed_hz": FIXED_HZ,
		"frames_per_clip": FRAMES_PER_CLIP,
		"family_count": 5,
		"chassis_count": chassis.size(),
		"motion_count": Creature.MOTIONS.size(),
		"clip_count": clips.size(),
		"total_frames": clips.size() * FRAMES_PER_CLIP,
		"total_cell_samples": byte_offset / 4,
		"motion_order": Creature.MOTIONS,
		"motion_specs": Creature.MOTION_SPECS,
		"quantization": {
			"format": "position-delta-u16le-biased-v1",
			"components": ["x", "y"],
			"scale": POSITION_SCALE,
			"bias": POSITION_BIAS,
			"clipped_values": clipped_values,
			"minimum_encoded": minimum_delta_q,
			"maximum_encoded": maximum_delta_q,
		},
		"control_contract": {
			"move": "unit-disk-vec2",
			"aim": "normalized-vec2",
			"attack": "unit-scalar",
			"utility": "unit-scalar",
			"external_event": ["none", "impact", "terminal"],
		},
		"contracts": {
			"morphology": "coordinate-conditioned-safe-scaffold-v1",
			"motion": "layered-cellular-motion-13x20-v1",
			"orientation": "vertical-locked-2.5d-v1",
			"binary": "clip-major-frame-major-cell-major-xy-u16le-v1",
		},
		"source": source,
		"corpus_identity_sha256": _sha256_text("|".join(corpus_identity)),
		"artifacts": {
			"motion_frames": {
				"path": "motion_frames.u16le",
				"bytes": byte_offset,
				"sha256": binary_sha256,
			},
		},
		"chassis": chassis,
		"clips": clips,
	}
	var manifest_path := staging.path_join("manifest.json")
	var manifest_file := FileAccess.open(manifest_path, FileAccess.WRITE)
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
		"total_cell_samples": manifest["total_cell_samples"],
		"binary_bytes": byte_offset,
		"binary_sha256": binary_sha256,
		"corpus_identity_sha256": manifest["corpus_identity_sha256"],
	}
