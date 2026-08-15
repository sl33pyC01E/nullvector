class_name NeuralWorld
extends Node2D

const Neural = preload("res://scripts/creature_stage/creature_neural.gd")

const CHUNK_SIZE := 512.0
const ACTIVE_RADIUS := 3
const BIOMES := ["verdant mire", "glass steppe", "spore forest", "iron reef", "phase desert"]
const BIOME_COLORS := [
	Color("#0a2423"), Color("#101d2b"), Color("#122819"),
	Color("#211c20"), Color("#1d1530")
]
const RESOURCE_COLORS := {
	"biomass": Color("#88ff6b"),
	"mineral": Color("#ffbd55"),
	"fluid": Color("#55dfff"),
	"spore": Color("#ff76c9"),
	"phase": Color("#ad7cff"),
}

var world_seed := 0
var chunks: Dictionary = {}
var societies: Dictionary = {}
var structures: Array[Dictionary] = []
var focus_position := Vector2.ZERO
var simulation_epoch := 0
var cohort_clock := 0.0


func configure(seed_value: int) -> void:
	world_seed = seed_value
	chunks.clear()
	societies.clear()
	structures.clear()
	ensure_chunks(Vector2.ZERO)
	queue_redraw()


func _chunk_seed(coord: Vector2i) -> int:
	return int(world_seed) ^ (coord.x * 73856093) ^ (coord.y * 19349663) ^ 0x4B1D5EED


func world_to_chunk(position: Vector2) -> Vector2i:
	return Vector2i(floori(position.x / CHUNK_SIZE), floori(position.y / CHUNK_SIZE))


func ensure_chunks(position: Vector2) -> void:
	focus_position = position
	var center := world_to_chunk(position)
	for cy in range(center.y - ACTIVE_RADIUS, center.y + ACTIVE_RADIUS + 1):
		for cx in range(center.x - ACTIVE_RADIUS, center.x + ACTIVE_RADIUS + 1):
			var coord := Vector2i(cx, cy)
			if not chunks.has(coord):
				chunks[coord] = _generate_chunk(coord)
	queue_redraw()


func _generate_chunk(coord: Vector2i) -> Dictionary:
	var seed := _chunk_seed(coord)
	var rng := RandomNumberGenerator.new()
	rng.seed = seed
	var origin := Vector2(coord) * CHUNK_SIZE
	var field := Neural.world_field(origin + Vector2.ONE * CHUNK_SIZE * 0.5, world_seed)
	var biome_id := _biome_from_field(field)
	var texture_nodes: Array = []
	for index in range(28):
		texture_nodes.append({
			"pos": origin + Vector2(rng.randf_range(0.0, CHUNK_SIZE), rng.randf_range(0.0, CHUNK_SIZE)),
			"radius": rng.randf_range(18.0, 95.0),
			"tone": rng.randf_range(-0.12, 0.14),
			"kind": rng.randi_range(0, 3),
		})
	var resources: Array = []
	var resource_count := 10 + int(field[0] * 13.0 + field[4] * 7.0)
	for index in range(resource_count):
		var type := _resource_type(field, rng.randf())
		resources.append({
			"id": "%d:%d:r%d" % [coord.x, coord.y, index],
			"pos": origin + Vector2(rng.randf_range(26.0, CHUNK_SIZE - 26.0), rng.randf_range(26.0, CHUNK_SIZE - 26.0)),
			"type": type,
			"amount": rng.randf_range(0.45, 1.0),
			"max_amount": 1.0,
			"phase": rng.randf_range(0.0, TAU),
		})
	var settlement: Dictionary = {}
	var settlement_roll := rng.randf()
	if settlement_roll < 0.09 + field[4] * 0.08:
		settlement = _generate_settlement(coord, origin, rng, field)
	return {
		"coord": coord,
		"seed": seed,
		"origin": origin,
		"field": field,
		"biome_id": biome_id,
		"biome": BIOMES[biome_id],
		"texture": texture_nodes,
		"resources": resources,
		"settlement": settlement,
		"cohorts": _generate_cohorts(rng, field),
		"epoch": simulation_epoch,
	}


func _biome_from_field(field: PackedFloat32Array) -> int:
	var scores := [
		field[0] + field[1] * 0.8,
		field[2] + (1.0 - field[1]) * 0.5,
		field[0] * 0.7 + field[1] + (1.0 - field[2]) * 0.4,
		field[4] * 1.3 + field[2] * 0.4,
		field[3] * 1.7,
	]
	var best := 0
	for index in range(1, scores.size()):
		if scores[index] > scores[best]:
			best = index
	return best


func _resource_type(field: PackedFloat32Array, roll: float) -> String:
	var weights := [field[0] * 1.2, field[4], field[1] * 0.8, field[0] * field[1], field[3] * 0.9]
	var total := 0.0
	for weight in weights:
		total += weight
	var cursor := roll * maxf(total, 0.001)
	for index in range(weights.size()):
		cursor -= weights[index]
		if cursor <= 0.0:
			return ["biomass", "mineral", "fluid", "spore", "phase"][index]
	return "biomass"


func _generate_cohorts(rng: RandomNumberGenerator, field: PackedFloat32Array) -> Array:
	var result: Array = []
	for family_id in range(5):
		var suitability := field[0]
		match family_id:
			1: suitability = field[0] * 0.8 + field[1] * 0.4
			2: suitability = field[0] + field[1]
			3: suitability = field[3] * 1.5
			4: suitability = field[4] * 1.3
		var population := maxi(0, roundi(rng.randf_range(-5.0, 18.0) * suitability))
		if population > 0:
			result.append({
				"family_id": family_id,
				"population": population,
				"energy": rng.randf_range(0.35, 0.9),
				"migration": Vector2(rng.randf_range(-1.0, 1.0), rng.randf_range(-1.0, 1.0)),
				"genome_pool": rng.randi(),
			})
	return result


func _generate_settlement(coord: Vector2i, origin: Vector2, rng: RandomNumberGenerator, field: PackedFloat32Array) -> Dictionary:
	var founder_family := rng.randi_range(0, 4)
	var society_id := "soc_%08x" % abs(_chunk_seed(coord))
	if not societies.has(society_id):
		societies[society_id] = _generate_society(society_id, founder_family, rng)
	var center := origin + Vector2(rng.randf_range(130.0, CHUNK_SIZE - 130.0), rng.randf_range(130.0, CHUNK_SIZE - 130.0))
	var districts: Array = []
	var district_count := rng.randi_range(4, 8)
	for index in range(district_count):
		var angle := TAU * float(index) / float(district_count) + rng.randf_range(-0.18, 0.18)
		var distance := rng.randf_range(34.0, 92.0)
		districts.append({
			"pos": center + Vector2.from_angle(angle) * distance,
			"radius": rng.randf_range(18.0, 34.0),
			"function": ["nest", "workshop", "farm", "reservoir", "shrine", "market", "reactor"][index % 7],
		})
	return {
		"id": "%s:%d:%d" % [society_id, coord.x, coord.y],
		"society_id": society_id,
		"center": center,
		"population": rng.randi_range(45, 220),
		"stores": {"biomass": rng.randf_range(20.0, 90.0), "mineral": rng.randf_range(10.0, 110.0)},
		"districts": districts,
		"need": ["food", "water", "minerals", "medicine", "knowledge"][rng.randi_range(0, 4)],
		"field_affinity": field,
	}


func _generate_society(society_id: String, founder_family: int, rng: RandomNumberGenerator) -> Dictionary:
	var starts := ["Ka", "Vell", "Myr", "Oss", "Tch", "Aru", "Null", "Sere", "Gho", "Prax"]
	var ends := ["eth", "ara", "uun", "ix", "fold", "spire", "hive", "vault", "kin", "bloom"]
	var name: String = str(starts[rng.randi_range(0, starts.size() - 1)]) + str(ends[rng.randi_range(0, ends.size() - 1)])
	var ethos_pool := ["symbiosis", "predation", "pilgrimage", "accumulation", "transfiguration", "memory", "construction", "purity"]
	var trait_pool := ["organ barter", "dream cartography", "living masonry", "ancestor grafting", "spore law", "machine adoption", "ritual mutation", "fluid prophecy"]
	return {
		"id": society_id,
		"name": name,
		"founder_family": founder_family,
		"ethos": ethos_pool[rng.randi_range(0, ethos_pool.size() - 1)],
		"trait": trait_pool[rng.randi_range(0, trait_pool.size() - 1)],
		"aggression": rng.randf_range(-0.65, 0.85),
		"curiosity": rng.randf_range(0.1, 1.0),
		"cohesion": rng.randf_range(0.2, 1.0),
		"relations": {},
		"history": ["Founded in epoch %d by %s descendants." % [simulation_epoch, Neural.FAMILIES[founder_family]]],
	}


func simulate_cohorts(delta: float) -> void:
	cohort_clock += delta
	if cohort_clock < 1.0:
		return
	var steps := floori(cohort_clock)
	cohort_clock -= float(steps)
	for _step in range(steps):
		simulation_epoch += 1
		for coord in chunks:
			var chunk: Dictionary = chunks[coord]
			for resource in chunk["resources"]:
				var field: PackedFloat32Array = chunk["field"]
				resource["amount"] = minf(float(resource["max_amount"]), float(resource["amount"]) + 0.001 + field[0] * 0.002)
			for cohort in chunk["cohorts"]:
				var population := int(cohort["population"])
				var energy := float(cohort["energy"])
				var growth := (energy - 0.46) * 0.015 * float(population)
				cohort["population"] = maxi(0, population + roundi(growth))
				cohort["energy"] = clampf(energy + randf_range(-0.015, 0.012), 0.1, 1.0)
			if not chunk["settlement"].is_empty():
				var city: Dictionary = chunk["settlement"]
				city["population"] = maxi(8, int(city["population"]) + (1 if simulation_epoch % 12 == 0 else 0))
	queue_redraw()


func nearest_resource(position: Vector2, accepted: Array, max_distance := 700.0) -> Dictionary:
	var best: Dictionary = {}
	var best_distance := max_distance
	var center := world_to_chunk(position)
	for cy in range(center.y - 1, center.y + 2):
		for cx in range(center.x - 1, center.x + 2):
			var chunk: Dictionary = chunks.get(Vector2i(cx, cy), {})
			for resource in chunk.get("resources", []):
				if float(resource.get("amount", 0.0)) <= 0.01 or (not accepted.is_empty() and str(resource["type"]) not in accepted):
					continue
				var distance := position.distance_to(resource["pos"])
				if distance < best_distance:
					best = resource
					best_distance = distance
	return best


func consume_resource(resource_id: String, amount: float) -> float:
	for coord in chunks:
		for resource in chunks[coord]["resources"]:
			if str(resource["id"]) == resource_id:
				var consumed := minf(float(resource["amount"]), amount)
				resource["amount"] = float(resource["amount"]) - consumed
				return consumed
	return 0.0


func nearest_settlement(position: Vector2, max_distance := 900.0) -> Dictionary:
	var best: Dictionary = {}
	var best_distance := max_distance
	for coord in chunks:
		var settlement: Dictionary = chunks[coord]["settlement"]
		if settlement.is_empty():
			continue
		var distance := position.distance_to(settlement["center"])
		if distance < best_distance:
			best = settlement
			best_distance = distance
	return best


func current_biome(position: Vector2) -> String:
	var chunk: Dictionary = chunks.get(world_to_chunk(position), {})
	return str(chunk.get("biome", "unresolved field"))


func build_structure(position: Vector2, family_id: int, kind: String, owner_id: String) -> Dictionary:
	var snapped := Vector2(roundf(position.x / 8.0) * 8.0, roundf(position.y / 8.0) * 8.0)
	var radius: float = {
		"shelter": 25.0,
		"scent_den": 21.0,
		"root_node": 27.0,
		"phase_anchor": 23.0,
		"sentry": 24.0,
	}.get(kind, 22.0)
	for structure in structures:
		if snapped.distance_to(structure["pos"]) < radius + float(structure["radius"]) + 10.0:
			return {}
	var city: Dictionary = nearest_settlement(snapped, 100.0)
	if not city.is_empty():
		return {}
	var record := {
		"id": "built:%s:%d" % [owner_id, structures.size()],
		"pos": snapped,
		"radius": radius,
		"family_id": family_id,
		"kind": kind,
		"owner_id": owner_id,
		"integrity": 1.0,
		"built_epoch": simulation_epoch,
	}
	structures.append(record)
	queue_redraw()
	return record


func structure_collision_force(position: Vector2, body_radius: float, family_id: int) -> Vector2:
	var force := Vector2.ZERO
	for structure in structures:
		if int(structure["family_id"]) == family_id:
			continue
		var delta := position - Vector2(structure["pos"])
		var distance := delta.length()
		var minimum := body_radius * 0.62 + float(structure["radius"])
		if distance < minimum:
			var structure_normal := delta / distance if distance > 0.001 else Vector2.RIGHT
			force += structure_normal * (minimum - distance)
	# Friendly traffic can pass through its own architecture. Foreign district
	# hulls collide broadly, so a single decorative cell can never shear a body.
	for coord in chunks:
		var settlement: Dictionary = chunks[coord]["settlement"]
		if settlement.is_empty():
			continue
		var society: Dictionary = societies.get(settlement["society_id"], {})
		if int(society.get("founder_family", -1)) == family_id:
			continue
		for district in settlement["districts"]:
			var delta := position - Vector2(district["pos"])
			var distance := delta.length()
			var minimum := body_radius * 0.55 + float(district["radius"])
			if distance < minimum:
				var district_normal := delta / distance if distance > 0.001 else Vector2.RIGHT
				force += district_normal * (minimum - distance)
	return force.limit_length(30.0)


func _draw() -> void:
	var center := world_to_chunk(focus_position)
	for cy in range(center.y - ACTIVE_RADIUS, center.y + ACTIVE_RADIUS + 1):
		for cx in range(center.x - ACTIVE_RADIUS, center.x + ACTIVE_RADIUS + 1):
			var coord := Vector2i(cx, cy)
			if not chunks.has(coord):
				continue
			_draw_chunk(chunks[coord])
	for structure in structures:
		if Vector2(structure["pos"]).distance_to(focus_position) < CHUNK_SIZE * float(ACTIVE_RADIUS + 1):
			_draw_structure(structure)


func _draw_chunk(chunk: Dictionary) -> void:
	var origin: Vector2 = chunk["origin"]
	var biome_id := int(chunk["biome_id"])
	var base: Color = BIOME_COLORS[biome_id]
	# A common substrate removes visible chunk seams. Biome color arrives as
	# overlapping neural-field pools that remain continuous across boundaries.
	draw_rect(Rect2(origin, Vector2.ONE * CHUNK_SIZE), Color("#071719"), true)
	draw_circle(origin + Vector2.ONE * CHUNK_SIZE * 0.5, CHUNK_SIZE * 0.72, Color(base, 0.58))
	for node in chunk["texture"]:
		var color := base.lightened(maxf(0.0, float(node["tone"]))).darkened(maxf(0.0, -float(node["tone"])))
		color.a = 0.32
		if int(node["kind"]) == 0:
			draw_circle(node["pos"], float(node["radius"]), color)
		elif int(node["kind"]) == 1:
			draw_arc(node["pos"], float(node["radius"]), 0.0, TAU, 24, color.lightened(0.1), 2.0)
		else:
			var p: Vector2 = node["pos"]
			var r := float(node["radius"])
			draw_line(p - Vector2(r, r * 0.3), p + Vector2(r, r * 0.3), color, 2.0)
	for resource in chunk["resources"]:
		var amount := float(resource["amount"])
		if amount <= 0.01:
			continue
		var pos: Vector2 = resource["pos"]
		var color: Color = RESOURCE_COLORS[str(resource["type"])]
		var radius := 3.0 + amount * 7.0
		draw_circle(pos + Vector2(0, 5), Vector2(radius * 1.5, radius * 0.45).length() * 0.45, Color(0.0, 0.0, 0.0, 0.35))
		draw_circle(pos, radius, Color(color, 0.18))
		draw_arc(pos, radius, 0.0, TAU * amount, 14, Color(color, 0.75), 1.5)
		draw_circle(pos, 1.4 + amount * 1.5, color.lightened(0.18))
	if not chunk["settlement"].is_empty():
		_draw_settlement(chunk["settlement"])


func _draw_settlement(settlement: Dictionary) -> void:
	var society: Dictionary = societies.get(settlement["society_id"], {})
	var family_id := int(society.get("founder_family", 0))
	var color: Color = Neural.FAMILY_COLORS[family_id]
	var center: Vector2 = settlement["center"]
	for district in settlement["districts"]:
		var pos: Vector2 = district["pos"]
		var radius := float(district["radius"])
		draw_line(center, pos, Color(color, 0.18), 7.0)
		draw_line(center, pos, Color(color, 0.42), 1.5)
		draw_circle(pos + Vector2(0, radius * 0.55), radius * 0.8, Color(0.0, 0.0, 0.0, 0.28))
		draw_circle(pos, radius, Color(color.darkened(0.72), 0.92))
		draw_arc(pos, radius, 0.0, TAU, 24, Color(color, 0.72), 2.0)
		draw_circle(pos, radius * 0.34, Color(color, 0.18))
	draw_circle(center, 31.0, Color(color.darkened(0.65), 0.98))
	draw_arc(center, 36.0, 0.0, TAU, 32, Color(color, 0.9), 2.0)
	draw_string(ThemeDB.fallback_font, center + Vector2(-55, -47), str(society.get("name", "settlement")), HORIZONTAL_ALIGNMENT_CENTER, 110, 11, color.lightened(0.25))


func _draw_structure(structure: Dictionary) -> void:
	var pos: Vector2 = structure["pos"]
	var radius := float(structure["radius"])
	var family_id := int(structure["family_id"])
	var color: Color = Neural.FAMILY_COLORS[family_id]
	var kind := str(structure["kind"])
	draw_circle(pos + Vector2(0, radius * 0.62), radius * 0.78, Color(0.0, 0.0, 0.0, 0.34))
	match kind:
		"root_node":
			for arm in range(6):
				var tip := pos + Vector2.from_angle(TAU * float(arm) / 6.0) * radius
				draw_line(pos, tip, Color(color, 0.65), 5.0)
			draw_circle(pos, radius * 0.48, color.darkened(0.45))
		"phase_anchor":
			draw_circle(pos, radius * 0.68, Color(color, 0.13))
			draw_arc(pos, radius, -simulation_epoch * 0.03, TAU - simulation_epoch * 0.03, 6, color, 2.0)
			draw_circle(pos, 4.0, color.lightened(0.25))
		"sentry":
			draw_rect(Rect2(pos - Vector2(radius * 0.58, radius * 0.48), Vector2(radius * 1.16, radius * 0.96)), color.darkened(0.58), true)
			draw_arc(pos, radius * 0.76, 0.0, TAU, 8, color, 2.0)
		"scent_den":
			draw_arc(pos, radius, PI, TAU, 18, color, 5.0)
			draw_circle(pos + Vector2(0, 4), radius * 0.48, color.darkened(0.68))
		_:
			draw_colored_polygon(PackedVector2Array([
				pos + Vector2(0, -radius), pos + Vector2(radius, 0),
				pos + Vector2(radius * 0.65, radius), pos + Vector2(-radius * 0.65, radius),
				pos + Vector2(-radius, 0),
			]), color.darkened(0.62))
			draw_polyline(PackedVector2Array([
				pos + Vector2(0, -radius), pos + Vector2(radius, 0), pos + Vector2(radius * 0.65, radius),
				pos + Vector2(-radius * 0.65, radius), pos + Vector2(-radius, 0), pos + Vector2(0, -radius),
			]), color, 2.0)
