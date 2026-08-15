class_name CreatureNeural
extends RefCounted

## Small native neural substrate used by the creature-stage reboot.
## The runtime supplies bounded tensor inference; generated fields and policies
## remain the authority over morphology, sensing, and action.

const FAMILIES := ["humanoid", "animalian", "plantlike", "anomaly", "machine"]
const FAMILY_COLORS := [
	Color("#4ce7ff"), Color("#ff6fb5"), Color("#9dff4f"),
	Color("#b789ff"), Color("#ffb13b")
]
const TISSUE_COLORS := {
	"skin": Color("#5fc9d7"),
	"structure": Color("#7895a4"),
	"armor": Color("#9aa8b8"),
	"neural": Color("#dc78ff"),
	"circulatory": Color("#ff4f72"),
	"respiratory": Color("#68e7ff"),
	"digestive": Color("#ffc85b"),
	"sensor": Color("#f4ffff"),
	"locomotor": Color("#77ff9d"),
	"storage": Color("#ffd886"),
	"phase": Color("#b184ff"),
	"root": Color("#95d45d"),
	"weapon": Color("#ff7658"),
}
const VITAL_ORGANS := [
	"brain", "heart", "lung", "gut", "eye",
	"meristem", "vascular", "photoreceptor",
	"phase_brain", "flux", "transmuter", "singularity",
	"processor", "coolant_pump", "radiator", "battery", "optic",
]


static func _sigmoid(value: float) -> float:
	return 1.0 / (1.0 + exp(-value))


static func _hash_noise(x: float, y: float, seed: int) -> float:
	# Continuous value-noise style feature. This is used as an input to neural
	# fields, never as the final world or morphology decision by itself.
	var value := sin(x * 12.9898 + y * 78.233 + float(seed % 1000003) * 0.000137)
	return sin(value * 43758.5453)


static func make_policy(family_id: int, seed: int) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed ^ 0x51A7C0DE ^ (family_id * 0x193D)
	var input_count := 18
	var hidden_count := 12
	var output_count := 10
	var wx: Array = []
	var wh: Array = []
	var bh: Array = []
	var wo: Array = []
	var bo: Array = []
	for h in range(hidden_count):
		var row: Array[float] = []
		for i in range(input_count):
			row.append(rng.randfn(0.0, 0.16))
		wx.append(row)
		var recurrent: Array[float] = []
		for j in range(hidden_count):
			recurrent.append(rng.randfn(0.0, 0.09))
		wh.append(recurrent)
		bh.append(rng.randfn(0.0, 0.05))
	for o in range(output_count):
		var row: Array[float] = []
		for h in range(hidden_count):
			row.append(rng.randfn(0.0, 0.13))
		wo.append(row)
		bo.append(rng.randfn(0.0, 0.04))

	# Install a readable evolved instinct scaffold into the recurrent network.
	# Inputs: food xy, prey xy, threat xy, mate xy, energy, health, local field,
	# crowding, age, day phase, noise, player xy, neural integrity.
	# Outputs: move xy, feed, attack, mate, utility, sprint, attention xy, rest.
	var appetites := [0.72, 1.15, 0.35, 0.38, 0.55]
	var aggression := [0.58, 0.92, -0.35, 0.42, 0.72]
	var fear := [0.45, 0.62, 0.18, -0.20, 0.28]
	var social := [0.72, 0.25, 0.62, -0.35, 0.42]
	var field_bias := [0.18, 0.08, 1.20, 1.35, 0.90]
	# Hidden pairs are interpretable latent drives.
	wx[0][0] += appetites[family_id]
	wx[0][2] += aggression[family_id]
	wx[0][4] -= fear[family_id]
	wx[0][6] += social[family_id] * 0.4
	wx[0][10] += field_bias[family_id] * 0.2
	wx[1][1] += appetites[family_id]
	wx[1][3] += aggression[family_id]
	wx[1][5] -= fear[family_id]
	wx[1][7] += social[family_id] * 0.4
	wx[1][10] += field_bias[family_id] * 0.2
	wx[2][8] -= 1.25
	wx[2][0] += 0.5
	wx[3][9] -= 1.1
	wx[4][11] -= 0.8
	wx[5][6] += social[family_id]
	wx[5][7] += social[family_id]
	wx[6][2] += aggression[family_id]
	wx[6][3] += aggression[family_id]
	wx[7][10] += field_bias[family_id]
	wh[0][0] += 0.32
	wh[1][1] += 0.32
	wh[2][2] += 0.55
	wh[3][3] += 0.52
	wh[4][4] += 0.38
	wh[5][5] += 0.48
	for h in range(hidden_count):
		wo[0][h] += (1.0 if h == 0 else 0.0)
		wo[1][h] += (1.0 if h == 1 else 0.0)
	wo[2][2] += 1.35
	wo[3][6] += 1.25
	wo[4][5] += 1.0
	wo[5][7] += 1.15
	wo[6][6] += 0.55
	wo[7][0] += 0.65
	wo[8][1] += 0.65
	wo[9][3] += 1.0
	return {
		"family_id": family_id,
		"seed": seed,
		"wx": wx, "wh": wh, "bh": bh, "wo": wo, "bo": bo,
		"hidden": PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
	}


static func policy_step(policy: Dictionary, inputs: PackedFloat32Array) -> PackedFloat32Array:
	var wx: Array = policy["wx"]
	var wh: Array = policy["wh"]
	var bh: Array = policy["bh"]
	var wo: Array = policy["wo"]
	var bo: Array = policy["bo"]
	var previous: PackedFloat32Array = policy["hidden"]
	var hidden := PackedFloat32Array()
	hidden.resize(wx.size())
	for h in range(wx.size()):
		var value := float(bh[h])
		for i in range(min(inputs.size(), wx[h].size())):
			value += float(wx[h][i]) * inputs[i]
		for j in range(previous.size()):
			value += float(wh[h][j]) * previous[j]
		hidden[h] = tanh(value)
	policy["hidden"] = hidden
	var outputs := PackedFloat32Array()
	outputs.resize(wo.size())
	for o in range(wo.size()):
		var value := float(bo[o])
		for h in range(hidden.size()):
			value += float(wo[o][h]) * hidden[h]
		outputs[o] = tanh(value)
	return outputs


static func mutate_policy(source: Dictionary, seed: int, amount := 0.08) -> Dictionary:
	var result: Dictionary = source.duplicate(true)
	var rng := RandomNumberGenerator.new()
	rng.seed = seed ^ 0x6D2B79F5
	for key in ["wx", "wh", "wo"]:
		var matrix: Array = result[key]
		for y in range(matrix.size()):
			for x in range(matrix[y].size()):
				if rng.randf() < 0.14:
					matrix[y][x] = float(matrix[y][x]) + rng.randfn(0.0, amount)
		result[key] = matrix
	result["seed"] = seed
	result["hidden"] = PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	return result


static func world_field(world_position: Vector2, world_seed: int) -> PackedFloat32Array:
	# A compact fixed MLP expressed analytically for very cheap dense sampling.
	# Outputs are smooth latent biome coordinates rather than authored tiles.
	var x := world_position.x * 0.00042
	var y := world_position.y * 0.00042
	var n0 := _hash_noise(floor(x * 7.0) / 7.0, floor(y * 7.0) / 7.0, world_seed)
	var h0 := tanh(sin(x * 2.1 + n0) * 1.4 + cos(y * 1.7) * 0.8)
	var h1 := tanh(cos(x * 1.2 - y * 1.9) + sin(y * 2.6 + n0 * 0.5))
	var h2 := tanh(sin((x + y) * 0.83) + cos((x - y) * 1.37))
	var h3 := tanh(h0 * 0.8 - h1 * 0.55 + h2 * 0.35 + n0 * 0.2)
	return PackedFloat32Array([
		_sigmoid(h0 * 2.0), # nutrient
		_sigmoid(h1 * 2.0), # moisture
		_sigmoid(h2 * 2.0), # temperature
		_sigmoid(h3 * 2.0), # anomaly flux
		_sigmoid((h0 + h1 - h2) * 1.2), # mineral
	])


static func _append_cell(cells: Array, seen: Dictionary, grid: Vector2i, tissue: String, organ: String, appendage: int = -1, side: int = 0) -> void:
	var key := "%d:%d" % [grid.x, grid.y]
	if seen.has(key):
		var existing: Dictionary = cells[int(seen[key])]
		if organ in VITAL_ORGANS or (existing.get("organ", "") == "none" and organ != "none"):
			existing["organ"] = organ
			existing["tissue"] = tissue
		return
	seen[key] = cells.size()
	cells.append({
		"grid": grid,
		"rest": Vector2(grid.x * 3.0, grid.y * 3.0),
		"pos": Vector2(grid.x * 3.0, grid.y * 3.0),
		"vel": Vector2.ZERO,
		"tissue": tissue,
		"organ": organ,
		"appendage": appendage,
		"side": side,
		"health": 1.0,
		"alive": true,
	})


static func _ellipse(cells: Array, seen: Dictionary, center: Vector2, radius: Vector2, tissue: String, organ := "none", appendage := -1, side := 0) -> void:
	var min_x := int(floor(center.x - radius.x))
	var max_x := int(ceil(center.x + radius.x))
	var min_y := int(floor(center.y - radius.y))
	var max_y := int(ceil(center.y + radius.y))
	for y in range(min_y, max_y + 1):
		for x in range(min_x, max_x + 1):
			var dx := (float(x) - center.x) / maxf(radius.x, 0.5)
			var dy := (float(y) - center.y) / maxf(radius.y, 0.5)
			if dx * dx + dy * dy <= 1.0:
				_append_cell(cells, seen, Vector2i(x, y), tissue, organ, appendage, side)


static func _line(cells: Array, seen: Dictionary, from: Vector2i, to: Vector2i, width: int, tissue: String, organ := "none", appendage := -1, side := 0) -> void:
	var steps := maxi(abs(to.x - from.x), abs(to.y - from.y))
	for step in range(steps + 1):
		var t := float(step) / maxf(1.0, float(steps))
		var point := Vector2(from).lerp(Vector2(to), t)
		for oy in range(-width, width + 1):
			for ox in range(-width, width + 1):
				if ox * ox + oy * oy <= width * width + 1:
					_append_cell(cells, seen, Vector2i(roundi(point.x) + ox, roundi(point.y) + oy), tissue, organ, appendage, side)


static func decode_morphology(family_id: int, seed: int, generation := 0) -> Dictionary:
	var rng := RandomNumberGenerator.new()
	rng.seed = seed ^ (generation * 0x45D9F3B)
	var cells: Array = []
	var seen: Dictionary = {}
	var asymmetry := rng.randf_range(-0.45, 0.45)
	var width_gene := rng.randf_range(0.86, 1.18)
	var height_gene := rng.randf_range(0.88, 1.16)
	match family_id:
		0: # Upright humanoid: crown, trunk, paired arms, separated legs.
			_ellipse(cells, seen, Vector2(0, -7), Vector2(3.2 * width_gene, 2.8), "skin")
			_ellipse(cells, seen, Vector2(0, -1), Vector2(3.7 * width_gene, 5.3 * height_gene), "skin")
			_line(cells, seen, Vector2i(-3, -3), Vector2i(-8, 2), 1, "locomotor", "none", 0, -1)
			_line(cells, seen, Vector2i(3, -3), Vector2i(8, 2), 1, "locomotor", "none", 1, 1)
			_line(cells, seen, Vector2i(-2, 3), Vector2i(-3, 10), 1, "locomotor", "none", 2, -1)
			_line(cells, seen, Vector2i(2, 3), Vector2i(3, 10), 1, "locomotor", "none", 3, 1)
			_ellipse(cells, seen, Vector2(0, -7), Vector2(1.8, 1.3), "neural", "brain")
			_ellipse(cells, seen, Vector2(0, -2), Vector2(1.3, 1.5), "circulatory", "heart")
			_ellipse(cells, seen, Vector2(-2, -1), Vector2(1.0, 1.5), "respiratory", "lung")
			_ellipse(cells, seen, Vector2(2, -1), Vector2(1.0, 1.5), "respiratory", "lung")
			_ellipse(cells, seen, Vector2(0, 2), Vector2(1.8, 1.8), "digestive", "gut")
			_append_cell(cells, seen, Vector2i(-1, -8), "sensor", "eye", 4, -1)
			_append_cell(cells, seen, Vector2i(1, -8), "sensor", "eye", 5, 1)
		1: # Animalian: broad body, four legs, muzzle/crown, tail.
			_ellipse(cells, seen, Vector2(0, 0), Vector2(7.4 * width_gene, 4.8 * height_gene), "skin")
			_ellipse(cells, seen, Vector2(0, -6), Vector2(4.5 * width_gene, 3.0), "skin")
			for limb in range(4):
				var side := -1 if limb % 2 == 0 else 1
				var root_x := -5 if limb < 2 else 5
				var foot_x := root_x + side * 2
				_line(cells, seen, Vector2i(root_x, 2), Vector2i(foot_x, 9), 1, "locomotor", "none", limb, side)
			_line(cells, seen, Vector2i(6, 0), Vector2i(11, 3 + roundi(asymmetry)), 1, "locomotor", "none", 4, 1)
			_ellipse(cells, seen, Vector2(0, -5), Vector2(2.0, 1.5), "neural", "brain")
			_ellipse(cells, seen, Vector2(-1.5, -1), Vector2(1.4, 1.4), "circulatory", "heart")
			_ellipse(cells, seen, Vector2(2, -1), Vector2(1.8, 1.4), "respiratory", "lung")
			_ellipse(cells, seen, Vector2(0, 2), Vector2(2.4, 1.7), "digestive", "gut")
			_append_cell(cells, seen, Vector2i(-2, -7), "sensor", "eye", 5, -1)
			_append_cell(cells, seen, Vector2i(2, -7), "sensor", "eye", 6, 1)
		2: # Plantlike: root plate, stem, crown/fronds, bulbs and runners.
			_ellipse(cells, seen, Vector2(0, 4), Vector2(5.2 * width_gene, 3.6), "root", "root")
			_line(cells, seen, Vector2i(0, 5), Vector2i(0, -7), 2, "structure", "stem")
			_ellipse(cells, seen, Vector2(0, -7), Vector2(3.3, 2.6), "storage", "bulb")
			for branch in range(3):
				var y := -2 - branch * 2
				var reach := 5 + branch
				_line(cells, seen, Vector2i(-1, y), Vector2i(-reach, y - 2), 1, "skin", "frond", branch * 2, -1)
				_line(cells, seen, Vector2i(1, y), Vector2i(reach, y - 2), 1, "skin", "frond", branch * 2 + 1, 1)
			_line(cells, seen, Vector2i(-3, 5), Vector2i(-10, 8), 1, "root", "runner", 6, -1)
			_line(cells, seen, Vector2i(3, 5), Vector2i(10, 8), 1, "root", "runner", 7, 1)
			_ellipse(cells, seen, Vector2(0, -5), Vector2(1.4, 1.3), "neural", "meristem")
			_ellipse(cells, seen, Vector2(0, 1), Vector2(1.5, 1.5), "circulatory", "vascular")
			_ellipse(cells, seen, Vector2(0, -7), Vector2(1.2, 1.1), "sensor", "photoreceptor")
		3: # Anomaly: central phase core, disconnected-looking orbital lobes.
			_ellipse(cells, seen, Vector2(0, 0), Vector2(3.6, 4.6), "phase", "core")
			for island in range(4):
				var side := -1 if island % 2 == 0 else 1
				var y := -5 if island < 2 else 5
				var x := side * (6 + island / 2)
				_ellipse(cells, seen, Vector2(x, y), Vector2(2.2, 1.8), "phase", "orbital", island, side)
				_line(cells, seen, Vector2i(signi(x) * 2, signi(y) * 2), Vector2i(x - signi(x) * 1, y - signi(y)), 0, "phase", "phase_bond", island, side)
			_ellipse(cells, seen, Vector2(0, 0), Vector2(1.4, 1.6), "neural", "phase_brain")
			_append_cell(cells, seen, Vector2i(0, -3), "sensor", "singularity", 5, 0)
			_append_cell(cells, seen, Vector2i(-2, 2), "circulatory", "flux")
			_append_cell(cells, seen, Vector2i(2, 2), "digestive", "transmuter")
		4: # Machine: armored box, lower tracks, mast and side hardpoints.
			_ellipse(cells, seen, Vector2(0, 0), Vector2(6.0 * width_gene, 5.0 * height_gene), "armor")
			_line(cells, seen, Vector2i(-5, 2), Vector2i(-5, 9), 2, "locomotor", "drive", 0, -1)
			_line(cells, seen, Vector2i(5, 2), Vector2i(5, 9), 2, "locomotor", "drive", 1, 1)
			_line(cells, seen, Vector2i(0, -3), Vector2i(0, -9), 1, "structure", "mast", 2, 0)
			_line(cells, seen, Vector2i(-4, -2), Vector2i(-9, -3), 1, "weapon", "hardpoint", 3, -1)
			_line(cells, seen, Vector2i(4, -2), Vector2i(9, -3), 1, "weapon", "hardpoint", 4, 1)
			_ellipse(cells, seen, Vector2(0, -2), Vector2(1.8, 1.6), "neural", "processor")
			_ellipse(cells, seen, Vector2(-2, 1), Vector2(1.3, 1.5), "circulatory", "coolant_pump")
			_ellipse(cells, seen, Vector2(2, 1), Vector2(1.3, 1.5), "respiratory", "radiator")
			_ellipse(cells, seen, Vector2(0, 3), Vector2(1.8, 1.5), "storage", "battery")
			_append_cell(cells, seen, Vector2i(0, -9), "sensor", "optic", 5, 0)

	# Neural micro-topology pass: family prior provides a safe scaffold while a
	# coordinate-conditioned network changes boundary occupancy and materials.
	var boundary_seed := seed ^ 0x2C1B3C6D
	for cell in cells:
		var grid: Vector2i = cell["grid"]
		var noise := _hash_noise(float(grid.x) * 0.31, float(grid.y) * 0.31, boundary_seed)
		if abs(noise) > 0.92 and cell["organ"] == "none" and abs(grid.x) > 2:
			cell["health"] = 0.82
		if noise > 0.78 and cell["tissue"] in ["skin", "structure"]:
			cell["tissue"] = "armor" if family_id == 4 else "skin"
	return {
		"family_id": family_id,
		"family": FAMILIES[family_id],
		"seed": seed,
		"generation": generation,
		"cells": cells,
		"genes": {
			"width": width_gene, "height": height_gene,
			"asymmetry": asymmetry,
			"repair": rng.randf_range(0.65, 1.35),
			"metabolism": rng.randf_range(0.72, 1.28),
			"fertility": rng.randf_range(0.65, 1.2),
			"bond_strength": rng.randf_range(0.78, 1.3),
		},
	}


static func tissue_color(tissue: String, family_id: int, health: float, pulse: float) -> Color:
	var base: Color = TISSUE_COLORS.get(tissue, FAMILY_COLORS[family_id])
	var family_tint: Color = FAMILY_COLORS[family_id]
	base = base.lerp(family_tint, 0.24)
	base = base.lerp(Color("#421629"), clampf(1.0 - health, 0.0, 0.75))
	if tissue in ["neural", "sensor", "phase", "weapon"]:
		base = base.lightened(0.10 + pulse * 0.12)
	return base
