class_name CreatureNeural
extends RefCounted

## Small native neural substrate used by the creature-stage reboot.
## The runtime supplies bounded tensor inference; generated fields and policies
## remain the authority over morphology, sensing, and action.

const FAMILIES := ["humanoid", "animalian", "plantlike", "anomaly", "machine"]
const MORPHOTYPES := [
	["balanced", "longarm", "sixlimb", "crowned"],
	["quadruped", "crawler", "longtail", "horned"],
	["treeform", "rosette", "runner", "twin_stem"],
	["triad", "cross", "pentad", "halo"],
	["tracked", "walker", "hover", "crab"],
]
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
	var morphotype_id: int = abs(seed ^ (generation * 0x9E37)) % 4
	var asymmetry := rng.randf_range(-0.18, 0.18)
	var width_gene := rng.randf_range(0.86, 1.18)
	var height_gene := rng.randf_range(0.88, 1.16)
	match family_id:
		0: # Upright humanoid: crown, trunk, paired arms, separated legs.
			var head_width: float = [3.2, 2.8, 3.5, 4.2][morphotype_id]
			var trunk_width: float = [3.7, 3.2, 4.4, 3.5][morphotype_id]
			var arm_reach: int = [8, 11, 8, 9][morphotype_id]
			var leg_spread: int = [3, 3, 5, 4][morphotype_id]
			_ellipse(cells, seen, Vector2(0, -7), Vector2(head_width * width_gene, 2.8), "skin")
			_ellipse(cells, seen, Vector2(0, -1), Vector2(trunk_width * width_gene, 5.3 * height_gene), "skin")
			_line(cells, seen, Vector2i(-3, -3), Vector2i(-arm_reach, 2), 1, "locomotor", "none", 0, -1)
			_line(cells, seen, Vector2i(3, -3), Vector2i(arm_reach, 2), 1, "locomotor", "none", 1, 1)
			_line(cells, seen, Vector2i(-2, 3), Vector2i(-leg_spread, 10), 1, "locomotor", "none", 2, -1)
			_line(cells, seen, Vector2i(2, 3), Vector2i(leg_spread, 10), 1, "locomotor", "none", 3, 1)
			if morphotype_id == 2:
				_line(cells, seen, Vector2i(-3, 0), Vector2i(-7, 5), 1, "locomotor", "none", 6, -1)
				_line(cells, seen, Vector2i(3, 0), Vector2i(7, 5), 1, "locomotor", "none", 7, 1)
			elif morphotype_id == 3:
				_line(cells, seen, Vector2i(-2, -8), Vector2i(-4, -11), 1, "sensor", "none", 6, -1)
				_line(cells, seen, Vector2i(2, -8), Vector2i(4, -11), 1, "sensor", "none", 7, 1)
			_ellipse(cells, seen, Vector2(0, -7), Vector2(1.8, 1.3), "neural", "brain")
			_ellipse(cells, seen, Vector2(0, -2), Vector2(1.3, 1.5), "circulatory", "heart")
			_ellipse(cells, seen, Vector2(-2, -1), Vector2(1.0, 1.5), "respiratory", "lung")
			_ellipse(cells, seen, Vector2(2, -1), Vector2(1.0, 1.5), "respiratory", "lung")
			_ellipse(cells, seen, Vector2(0, 2), Vector2(1.8, 1.8), "digestive", "gut")
			_append_cell(cells, seen, Vector2i(-1, -8), "sensor", "eye", 4, -1)
			_append_cell(cells, seen, Vector2i(1, -8), "sensor", "eye", 5, 1)
		1: # Animalian: broad body, four legs, muzzle/crown, tail.
			var body_width: float = [7.4, 9.2, 6.8, 8.0][morphotype_id]
			var body_height: float = [4.8, 3.8, 5.1, 4.5][morphotype_id]
			_ellipse(cells, seen, Vector2(0, 0), Vector2(body_width * width_gene, body_height * height_gene), "skin")
			_ellipse(cells, seen, Vector2(0, -6), Vector2(4.5 * width_gene, 3.0), "skin")
			var leg_pairs := 3 if morphotype_id == 1 else 2
			var appendage_index := 0
			for pair in range(leg_pairs):
				var root_y := -1 + pair * 3
				var root_x := roundi(body_width * 0.68)
				_line(cells, seen, Vector2i(-root_x, root_y), Vector2i(-root_x - 2, 9 + pair), 1, "locomotor", "none", appendage_index, -1)
				appendage_index += 1
				_line(cells, seen, Vector2i(root_x, root_y), Vector2i(root_x + 2, 9 + pair), 1, "locomotor", "none", appendage_index, 1)
				appendage_index += 1
			var tail_reach := 15 if morphotype_id == 2 else 11
			_line(cells, seen, Vector2i(roundi(body_width - 1.0), 0), Vector2i(tail_reach, 3 + roundi(asymmetry)), 1, "locomotor", "none", appendage_index, 1)
			if morphotype_id == 3:
				_line(cells, seen, Vector2i(-2, -7), Vector2i(-4, -11), 1, "weapon", "none", appendage_index + 1, -1)
				_line(cells, seen, Vector2i(2, -7), Vector2i(4, -11), 1, "weapon", "none", appendage_index + 2, 1)
			_ellipse(cells, seen, Vector2(0, -5), Vector2(2.0, 1.5), "neural", "brain")
			_ellipse(cells, seen, Vector2(-1.5, -1), Vector2(1.4, 1.4), "circulatory", "heart")
			_ellipse(cells, seen, Vector2(2, -1), Vector2(1.8, 1.4), "respiratory", "lung")
			_ellipse(cells, seen, Vector2(0, 2), Vector2(2.4, 1.7), "digestive", "gut")
			_append_cell(cells, seen, Vector2i(-2, -7), "sensor", "eye", 5, -1)
			_append_cell(cells, seen, Vector2i(2, -7), "sensor", "eye", 6, 1)
		2: # Plantlike: root plate, stem, crown/fronds, bulbs and runners.
			var root_width: float = [5.2, 7.2, 5.8, 6.0][morphotype_id]
			var stem_top: int = [-7, -5, -8, -9][morphotype_id]
			_ellipse(cells, seen, Vector2(0, 4), Vector2(root_width * width_gene, 3.6), "root", "root")
			_line(cells, seen, Vector2i(0, 5), Vector2i(0, stem_top), 2, "structure", "stem")
			_ellipse(cells, seen, Vector2(0, -7), Vector2(3.3, 2.6), "storage", "bulb")
			var branch_pairs: int = [3, 4, 2, 4][morphotype_id]
			for branch in range(branch_pairs):
				var y := -2 - branch * 2
				var reach := (7 if morphotype_id == 1 else 5) + branch
				_line(cells, seen, Vector2i(-1, y), Vector2i(-reach, y - 2), 1, "skin", "frond", branch * 2, -1)
				_line(cells, seen, Vector2i(1, y), Vector2i(reach, y - 2), 1, "skin", "frond", branch * 2 + 1, 1)
			var runner_reach := 14 if morphotype_id == 2 else 10
			_line(cells, seen, Vector2i(-3, 5), Vector2i(-runner_reach, 8), 1, "root", "runner", branch_pairs * 2, -1)
			_line(cells, seen, Vector2i(3, 5), Vector2i(runner_reach, 8), 1, "root", "runner", branch_pairs * 2 + 1, 1)
			if morphotype_id == 2:
				_line(cells, seen, Vector2i(-2, 6), Vector2i(-8, 12), 1, "root", "runner", branch_pairs * 2 + 2, -1)
				_line(cells, seen, Vector2i(2, 6), Vector2i(8, 12), 1, "root", "runner", branch_pairs * 2 + 3, 1)
			elif morphotype_id == 3:
				_line(cells, seen, Vector2i(-2, 2), Vector2i(-3, -8), 1, "structure", "stem", branch_pairs * 2 + 2, -1)
				_line(cells, seen, Vector2i(2, 2), Vector2i(3, -8), 1, "structure", "stem", branch_pairs * 2 + 3, 1)
				_ellipse(cells, seen, Vector2(-3, -8), Vector2(2.0, 1.8), "storage", "bulb")
				_ellipse(cells, seen, Vector2(3, -8), Vector2(2.0, 1.8), "storage", "bulb")
			_ellipse(cells, seen, Vector2(0, -5), Vector2(1.4, 1.3), "neural", "meristem")
			_ellipse(cells, seen, Vector2(0, 1), Vector2(1.5, 1.5), "circulatory", "vascular")
			_ellipse(cells, seen, Vector2(0, -7), Vector2(1.2, 1.1), "sensor", "photoreceptor")
		3: # Anomaly: central phase core, disconnected-looking orbital lobes.
			_ellipse(cells, seen, Vector2(0, 0), Vector2(3.6 * width_gene, 4.6 * height_gene), "phase", "core")
			var island_count: int = [3, 4, 5, 6][morphotype_id]
			var orbital_radius: float = [7.0, 7.4, 8.0, 9.0][morphotype_id]
			for island in range(island_count):
				var angle := -PI * 0.5 + TAU * float(island) / float(island_count)
				var orbital: Vector2 = Vector2(cos(angle), sin(angle)) * orbital_radius
				var x := roundi(orbital.x)
				var y := roundi(orbital.y)
				var side := signi(x)
				_ellipse(cells, seen, Vector2(x, y), Vector2(2.2, 1.8), "phase", "orbital", island, side)
				_line(cells, seen, Vector2i(roundi(cos(angle) * 2.0), roundi(sin(angle) * 2.0)), Vector2i(roundi(cos(angle) * (orbital_radius - 1.0)), roundi(sin(angle) * (orbital_radius - 1.0))), 1, "phase", "phase_bond", island, side)
			_ellipse(cells, seen, Vector2(0, 0), Vector2(1.4, 1.6), "neural", "phase_brain")
			_append_cell(cells, seen, Vector2i(0, -3), "sensor", "singularity", 5, 0)
			_append_cell(cells, seen, Vector2i(-2, 2), "circulatory", "flux")
			_append_cell(cells, seen, Vector2i(2, 2), "digestive", "transmuter")
		4: # Machine: armored box, lower tracks, mast and side hardpoints.
			var chassis_width: float = [6.0, 5.4, 7.6, 6.8][morphotype_id]
			var chassis_height: float = [5.0, 5.8, 4.1, 4.7][morphotype_id]
			_ellipse(cells, seen, Vector2(0, 0), Vector2(chassis_width * width_gene, chassis_height * height_gene), "armor")
			if morphotype_id == 0:
				_line(cells, seen, Vector2i(-5, 2), Vector2i(-5, 9), 2, "locomotor", "drive", 0, -1)
				_line(cells, seen, Vector2i(5, 2), Vector2i(5, 9), 2, "locomotor", "drive", 1, 1)
			elif morphotype_id == 1:
				for leg in range(2):
					var leg_y := 1 + leg * 3
					_line(cells, seen, Vector2i(-4, leg_y), Vector2i(-7, 10 + leg), 1, "locomotor", "drive", leg * 2, -1)
					_line(cells, seen, Vector2i(4, leg_y), Vector2i(7, 10 + leg), 1, "locomotor", "drive", leg * 2 + 1, 1)
			elif morphotype_id == 2:
				_ellipse(cells, seen, Vector2(-7, 5), Vector2(2.2, 2.2), "locomotor", "drive", 0, -1)
				_ellipse(cells, seen, Vector2(7, 5), Vector2(2.2, 2.2), "locomotor", "drive", 1, 1)
				_ellipse(cells, seen, Vector2(0, 7), Vector2(2.7, 1.7), "locomotor", "drive", 2, 0)
				_line(cells, seen, Vector2i(-4, 3), Vector2i(-7, 5), 1, "locomotor", "drive", 0, -1)
				_line(cells, seen, Vector2i(4, 3), Vector2i(7, 5), 1, "locomotor", "drive", 1, 1)
				_line(cells, seen, Vector2i(0, 3), Vector2i(0, 7), 1, "locomotor", "drive", 2, 0)
			else:
				for leg in range(3):
					var leg_y := -1 + leg * 3
					_line(cells, seen, Vector2i(-5, leg_y), Vector2i(-10, 6 + leg * 2), 1, "locomotor", "drive", leg * 2, -1)
					_line(cells, seen, Vector2i(5, leg_y), Vector2i(10, 6 + leg * 2), 1, "locomotor", "drive", leg * 2 + 1, 1)
			_line(cells, seen, Vector2i(0, -3), Vector2i(0, -9), 1, "structure", "mast", 2, 0)
			var hardpoint_reach := 12 if morphotype_id == 2 else 9
			_line(cells, seen, Vector2i(-4, -2), Vector2i(-hardpoint_reach, -3), 1, "weapon", "hardpoint", 6, -1)
			_line(cells, seen, Vector2i(4, -2), Vector2i(hardpoint_reach, -3), 1, "weapon", "hardpoint", 7, 1)
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
		"morphotype_id": morphotype_id,
		"morphotype": MORPHOTYPES[family_id][morphotype_id],
		"seed": seed,
		"generation": generation,
		"cells": cells,
		"genes": {
			"width": width_gene, "height": height_gene,
			"asymmetry": asymmetry,
			"symmetry": _symmetry_score(cells),
			"repair": rng.randf_range(0.65, 1.35),
			"metabolism": rng.randf_range(0.72, 1.28),
			"fertility": rng.randf_range(0.65, 1.2),
			"bond_strength": rng.randf_range(0.78, 1.3),
		},
	}


static func _symmetry_score(cells: Array) -> float:
	var occupied: Dictionary = {}
	for cell in cells:
		var grid: Vector2i = cell["grid"]
		occupied["%d:%d" % [grid.x, grid.y]] = true
	var mirrored := 0
	for cell in cells:
		var grid: Vector2i = cell["grid"]
		if occupied.has("%d:%d" % [-grid.x, grid.y]):
			mirrored += 1
	return float(mirrored) / float(maxi(cells.size(), 1))


static func analyze_morphology(blueprint: Dictionary) -> Dictionary:
	var cells: Array = blueprint["cells"]
	var occupied: Dictionary = {}
	var appendages: Dictionary = {}
	var organs: Dictionary = {}
	var min_grid := Vector2i(999, 999)
	var max_grid := Vector2i(-999, -999)
	var sensory_y := 0.0
	var sensory_count := 0
	var locomotor_y := 0.0
	var locomotor_count := 0
	for cell in cells:
		var grid: Vector2i = cell["grid"]
		occupied["%d:%d" % [grid.x, grid.y]] = grid
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
		if str(cell.get("tissue", "")) == "sensor" or organ in ["eye", "photoreceptor", "singularity", "optic"]:
			sensory_y += float(grid.y)
			sensory_count += 1
		if str(cell.get("tissue", "")) in ["locomotor", "root"] or (int(blueprint["family_id"]) == 3 and appendage >= 0):
			locomotor_y += float(grid.y)
			locomotor_count += 1
	var queue: Array[Vector2i] = []
	var visited: Dictionary = {}
	if not cells.is_empty():
		queue.append(cells[0]["grid"])
	while not queue.is_empty():
		var current: Vector2i = queue.pop_front()
		var key := "%d:%d" % [current.x, current.y]
		if visited.has(key):
			continue
		visited[key] = true
		for direction in [Vector2i.LEFT, Vector2i.RIGHT, Vector2i.UP, Vector2i.DOWN]:
			var neighbor: Vector2i = current + direction
			var neighbor_key := "%d:%d" % [neighbor.x, neighbor.y]
			if occupied.has(neighbor_key) and not visited.has(neighbor_key):
				queue.append(neighbor)
	var sensory_mean := sensory_y / float(maxi(sensory_count, 1))
	var locomotor_mean := locomotor_y / float(maxi(locomotor_count, 1))
	return {
		"family": blueprint["family"],
		"family_id": blueprint["family_id"],
		"morphotype": blueprint["morphotype"],
		"morphotype_id": blueprint["morphotype_id"],
		"cell_count": cells.size(),
		"width": max_grid.x - min_grid.x + 1,
		"height": max_grid.y - min_grid.y + 1,
		"appendage_count": appendages.size(),
		"organ_count": organs.size(),
		"organs": organs.keys(),
		"symmetry": _symmetry_score(cells),
		"connected_fraction": float(visited.size()) / float(maxi(cells.size(), 1)),
		"sensory_mean_y": sensory_mean,
		"locomotor_mean_y": locomotor_mean,
		"vertical_ordered": sensory_count > 0 and locomotor_count > 0 and sensory_mean < locomotor_mean,
		"signature": "%dx%d:c%d:a%d:o%d" % [max_grid.x - min_grid.x + 1, max_grid.y - min_grid.y + 1, cells.size(), appendages.size(), organs.size()],
	}


static func tissue_color(tissue: String, family_id: int, health: float, pulse: float) -> Color:
	var base: Color = TISSUE_COLORS.get(tissue, FAMILY_COLORS[family_id])
	var family_tint: Color = FAMILY_COLORS[family_id]
	base = base.lerp(family_tint, 0.24)
	base = base.lerp(Color("#421629"), clampf(1.0 - health, 0.0, 0.75))
	if tissue in ["neural", "sensor", "phase", "weapon"]:
		base = base.lightened(0.10 + pulse * 0.12)
	return base
