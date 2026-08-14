extends Node2D

const HUB_FORMAT := "nullvector-native-forge-hub-report-v1"
const TEXT := Color("#eaf8ff")
const MUTED := Color("#7890a6")
const DEEP := Color("#02060d")
const PANEL := Color("#06101d")
const RULE := Color("#1c4260")
const CYAN := Color("#37f3ff")
const PINK := Color("#ff4fb7")
const LIME := Color("#a8ff4f")
const VIOLET := Color("#a77bff")
const ORANGE := Color("#ffae37")
const RED := Color("#ff526d")

const FILTERS := ["all", "core", "neural", "motion", "organism", "ecology", "map"]
const LABS := [
	{
		"id": "arena", "title": "ARENA", "category": "core",
		"scene": "res://Arena.tscn", "manifest": "", "format": "",
		"description": "Top-down neon combat runtime and protected project entrypoint.",
		"metric": "LIVE", "metric_label": "TOP-DOWN GAME"
	},
	{
		"id": "forge_lab", "title": "FORGE LAB", "category": "core",
		"scene": "res://ForgeLab.tscn", "manifest": "res://generated/v2/asset_index.json",
		"format": "nullvector-forge-lab-assets-v1", "requires_ready": false,
		"metric_path": "motion.clip_count", "metric_expected": 520,
		"description": "Compact static, motion, and topology-v2 runtime asset inspector.",
		"metric": "520", "metric_label": "MOTION CLIPS"
	},
	{
		"id": "neural_workshop", "title": "NEURAL WORKSHOP", "category": "neural",
		"scene": "res://NeuralWorkshop.tscn", "manifest": "res://generated/neural_workshop/v1/asset_index.json",
		"format": "nullvector-neural-workshop-assets-v1",
		"metric_path": "static.identity_count", "metric_expected": 80,
		"description": "Actual neural identities, seven presentation layers, motion, and maps.",
		"metric": "80", "metric_label": "NEURAL IDENTITIES"
	},
	{
		"id": "neural_genetics", "title": "NEURAL GENETICS", "category": "neural",
		"scene": "res://NeuralGeneticsWorkshop.tscn", "manifest": "res://generated/neural_genetics/v3/asset_index.json",
		"format": "nullvector-neural-genetics-workshop-assets-v3",
		"metric_path": "asset_count", "metric_expected": 407,
		"description": "Latent walks, phenotype fusion, mutation, selection, and lineages.",
		"metric": "407", "metric_label": "HASHED ASSETS"
	},
	{
		"id": "repaired_motion", "title": "ALL-80 MOTION", "category": "motion",
		"scene": "res://RepairedMotionLab.tscn", "manifest": "res://generated/repaired_motion_lab/v1/catalog.json",
		"format": "nullvector-repaired-motion-native-catalog-v1",
		"metric_path": "counts.frame_count", "metric_expected": 75520,
		"description": "Every neural identity animated across 13 actions and eight facings.",
		"metric": "75,520", "metric_label": "ADDRESSABLE FRAMES"
	},
	{
		"id": "subtype_motion", "title": "SUBTYPE MOTION", "category": "motion",
		"scene": "res://SubtypeMotionLab.tscn", "manifest": "res://generated/morphology_subtype_lab/v1/catalog.json",
		"format": "nullvector-native-morphology-subtype-runtime-v1",
		"metric_path": "counts.identity_count", "metric_expected": 20,
		"description": "One procedural reference identity for every morphology subtype.",
		"metric": "20", "metric_label": "SUBTYPE IDENTITIES"
	},
	{
		"id": "cellular_motion", "title": "CELLULAR MOTION", "category": "motion",
		"scene": "res://CellularMotionLab.tscn", "manifest": "res://generated/cellular_motion/v12/motion_catalog.json",
		"format": "nullvector-cellular-neuromuscular-native-catalog-v7",
		"metric_path": "frame_count", "metric_expected": 4720,
		"description": "Bond-preserving organ motion with articulated appendage chains.",
		"metric": "4,720", "metric_label": "CELLULAR POSES"
	},
	{
		"id": "cellular_organism", "title": "CELLULAR ORGANISMS", "category": "organism",
		"scene": "res://CellularOrganismLab.tscn", "manifest": "res://generated/cellular_organism/v2/catalog.json",
		"format": "nullvector-cellular-organism-native-catalog-v1",
		"metric_path": "totals.physical_cells", "metric_expected": 34178,
		"description": "Physical cells, organs, fluids, appendages, eyes, and bond graphs.",
		"metric": "34,178", "metric_label": "PHYSICAL CELLS"
	},
	{
		"id": "symmetry", "title": "SYMMETRIC ORGANISMS", "category": "organism",
		"scene": "res://SymmetricOrganismLab.tscn", "manifest": "res://generated/cellular_symmetry/v1/catalog.json",
		"format": "nullvector-cellular-organism-native-catalog-v1",
		"metric_path": "symmetry_summary.improved_samples", "metric_expected": 45,
		"description": "Soft bilateral chassis and appendage pairing without flattening diversity.",
		"metric": "45/45", "metric_label": "SYMMETRY IMPROVED"
	},
	{
		"id": "evolved", "title": "EVOLVED ORGANISMS", "category": "organism",
		"scene": "res://EvolvedOrganismLab.tscn", "manifest": "res://generated/evolved_cellular_organism/v1/catalog.json",
		"format": "nullvector-cellular-organism-native-catalog-v1",
		"metric_path": "sample_count", "metric_expected": 36,
		"description": "Three generations of heritable cellular morphology evolution.",
		"metric": "3", "metric_label": "GENERATIONS"
	},
	{
		"id": "breeding", "title": "BREEDING LAB", "category": "organism",
		"scene": "res://CellularBreedingLab.tscn", "manifest": "res://generated/cellular_breeding/v1/catalog.json",
		"format": "nullvector-cellular-organism-native-catalog-v1",
		"metric_path": "sample_count", "metric_expected": 45,
		"description": "Cross-family fusion, mutation pixels, repair pixels, and inheritance.",
		"metric": "45", "metric_label": "HYBRID OFFSPRING"
	},
	{
		"id": "ecology", "title": "CELLULAR ECOLOGY", "category": "ecology",
		"scene": "res://CellularEcologyLab.tscn", "manifest": "res://generated/cellular_ecology/v6/ecology_catalog.json",
		"format": "nullvector-cellular-ecology-native-catalog-v3",
		"metric_path": "resource_node_count", "metric_expected": 120,
		"description": "Resources, trophic roles, predation, inorganic feeding, and powers.",
		"metric": "120", "metric_label": "RESOURCE NODES"
	},
	{
		"id": "ontogeny", "title": "CELLULAR ONTOGENY", "category": "ecology",
		"scene": "res://CellularOntogenyLab.tscn", "manifest": "res://generated/cellular_ontogeny/v6/ontogeny_catalog.json",
		"format": "nullvector-cellular-ontogeny-native-catalog-v3",
		"metric_path": "program_count", "metric_expected": 45,
		"description": "Cell lineage programs, developmental stages, pairing, and growth.",
		"metric": "45", "metric_label": "GROWTH PROGRAMS"
	},
	{
		"id": "neural_maps", "title": "NEURAL MAPS", "category": "map",
		"scene": "res://NeuralDecoratedMapLab.tscn", "manifest": "res://generated/neural_decorated_maps/v1_1/catalog.json",
		"format": "nullvector-neural-decorated-map-native-catalog/1.0.0",
		"metric_path": "atlas_frame_count", "metric_expected": 90,
		"description": "Six deterministic themes with neural decoration and topology masks.",
		"metric": "90", "metric_label": "MAP ATLAS FRAMES"
	}
]

var inventory: Array[Dictionary] = []
var inventory_errors: Array[String] = []
var cards: Array[Dictionary] = []
var selected_filter := "all"
var card_surface: Control
var status_label: Label
var count_label: Label
var filter_buttons: Dictionary = {}


func _ready() -> void:
	texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	inventory = _validate_inventory()
	_build_interface()
	_apply_filter("all")
	if "--forge-hub-smoke" in OS.get_cmdline_user_args():
		call_deferred("_run_headless_smoke")


func _draw() -> void:
	draw_rect(Rect2(0, 0, 1280, 720), DEEP)
	for x in range(0, 1281, 32):
		draw_line(Vector2(x, 0), Vector2(x, 720), Color(0.12, 0.45, 0.60, 0.07), 1.0)
	for y in range(0, 721, 32):
		draw_line(Vector2(0, y), Vector2(1280, y), Color(0.12, 0.45, 0.60, 0.07), 1.0)
	draw_circle(Vector2(1110, 102), 280.0, Color(0.12, 0.80, 1.0, 0.028))
	draw_circle(Vector2(120, 650), 250.0, Color(1.0, 0.18, 0.60, 0.026))
	draw_line(Vector2(0, 91), Vector2(1280, 91), Color(0.22, 0.95, 1.0, 0.24), 1.0)


func _validate_inventory() -> Array[Dictionary]:
	var records: Array[Dictionary] = []
	for lab in LABS:
		var errors: Array[String] = []
		var scene_path := str(lab.get("scene", ""))
		if not ResourceLoader.exists(scene_path):
			errors.append("scene missing")
		else:
			var scene_resource = ResourceLoader.load(scene_path, "PackedScene", ResourceLoader.CACHE_MODE_IGNORE)
			if not scene_resource is PackedScene:
				errors.append("scene invalid")
		var manifest_path := str(lab.get("manifest", ""))
		var manifest: Dictionary = {}
		var manifest_sha := "none"
		if not manifest_path.is_empty():
			manifest = _load_json(manifest_path)
			if manifest.is_empty():
				errors.append("manifest missing or invalid")
			else:
				manifest_sha = _sha256_file(manifest_path)
				if str(manifest.get("format", "")) != str(lab.get("format", "")):
					errors.append("manifest format")
				if bool(lab.get("requires_ready", true)) and str(manifest.get("status", "")) != "ready":
					errors.append("manifest status")
				if manifest.has("python_runtime_required") and bool(manifest.get("python_runtime_required", true)):
					errors.append("Python runtime required")
				if manifest.has("errors") and not manifest.get("errors", []).is_empty():
					errors.append("manifest errors")
				var bundle := str(manifest.get("bundle_id", ""))
				if not bundle.is_empty() and not _is_sha256(bundle):
					errors.append("bundle id")
				var metric_path := str(lab.get("metric_path", ""))
				if not metric_path.is_empty():
					var actual = _nested_value(manifest, metric_path)
					if actual != lab.get("metric_expected"):
						errors.append("metric " + metric_path)
		var record := {
			"id": str(lab.get("id", "")),
			"title": str(lab.get("title", "")),
			"category": str(lab.get("category", "")),
			"scene": scene_path,
			"scene_sha256": _sha256_file(scene_path) if FileAccess.file_exists(scene_path) else "missing",
			"manifest": manifest_path,
			"manifest_sha256": manifest_sha,
			"valid": errors.is_empty(),
			"errors": errors,
		}
		records.append(record)
		for error in errors:
			inventory_errors.append("%s: %s" % [str(lab.get("id", "unknown")), error])
	return records


func _build_interface() -> void:
	var canvas := CanvasLayer.new()
	add_child(canvas)
	var title := _label(canvas, Vector2(26, 13), Vector2(700, 36), "NULLVECTOR // NATIVE FORGE HUB", TEXT, 24)
	title.add_theme_color_override("font_shadow_color", Color(0.0, 0.9, 1.0, 0.24))
	_label(canvas, Vector2(28, 49), Vector2(720, 22), "SPRITES  x  MOTION  x  ORGANISMS  x  ECOLOGY  x  MAPS", CYAN, 10)
	status_label = _label(canvas, Vector2(740, 18), Vector2(510, 26), "", LIME if inventory_errors.is_empty() else RED, 10)
	status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	status_label.text = "%d LABS ONLINE // ARENA MAIN PRESERVED" % LABS.size() if inventory_errors.is_empty() else "FAIL-CLOSED // %d INVENTORY ERRORS" % inventory_errors.size()
	count_label = _label(canvas, Vector2(1005, 55), Vector2(245, 22), "", MUTED, 9)
	count_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT

	var filter_x := 26.0
	for filter_name in FILTERS:
		var color := _category_color(filter_name)
		var button := _button(canvas, Vector2(filter_x, 99), Vector2(126, 31), filter_name.to_upper(), _apply_filter.bind(filter_name), color)
		filter_buttons[filter_name] = button
		filter_x += 132.0

	var scroll := ScrollContainer.new()
	scroll.position = Vector2(24, 142)
	scroll.size = Vector2(1232, 526)
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
	scroll.vertical_scroll_mode = ScrollContainer.SCROLL_MODE_AUTO
	canvas.add_child(scroll)
	card_surface = Control.new()
	card_surface.custom_minimum_size = Vector2(1214, 620)
	scroll.add_child(card_surface)
	for index in range(LABS.size()):
		_build_card(index, LABS[index], inventory[index])

	var footer := _label(canvas, Vector2(24, 682), Vector2(1232, 22), "OPEN loads in this instance  //  DETACH preserves the hub  //  ESC closes the hub  //  Arena remains project main", MUTED, 9)
	footer.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER


func _build_card(index: int, lab: Dictionary, record: Dictionary) -> void:
	var panel := Panel.new()
	panel.size = Vector2(392, 146)
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.018, 0.05, 0.09, 0.96)
	style.border_color = _category_color(str(lab.get("category", "core"))) if bool(record.get("valid", false)) else RED
	style.set_border_width_all(1)
	style.corner_radius_top_left = 4
	style.corner_radius_top_right = 4
	style.corner_radius_bottom_left = 4
	style.corner_radius_bottom_right = 4
	panel.add_theme_stylebox_override("panel", style)
	card_surface.add_child(panel)

	var color := _category_color(str(lab.get("category", "core")))
	var ordinal := _label(panel, Vector2(13, 9), Vector2(42, 20), "%02d" % (index + 1), color, 9)
	ordinal.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	var title := _label(panel, Vector2(58, 8), Vector2(240, 23), str(lab.get("title", "LAB")), TEXT, 15)
	_label(panel, Vector2(304, 10), Vector2(75, 18), str(lab.get("category", "")).to_upper(), color, 8).horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	var description := _label(panel, Vector2(14, 39), Vector2(364, 36), str(lab.get("description", "")), MUTED, 9)
	description.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	var metric := _label(panel, Vector2(14, 80), Vector2(120, 27), str(lab.get("metric", "--")), color, 20)
	metric.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	_label(panel, Vector2(140, 88), Vector2(152, 18), str(lab.get("metric_label", "CONTRACT")), MUTED, 7)
	var proof := str(record.get("manifest_sha256", record.get("scene_sha256", "missing")))
	_label(panel, Vector2(14, 113), Vector2(210, 18), "SHA // " + proof.substr(0, 14), Color(color, 0.72), 7)
	var open_button := _button(panel, Vector2(226, 109), Vector2(74, 27), "OPEN", _open_lab.bind(str(lab.get("scene", ""))), color)
	var detach_button := _button(panel, Vector2(305, 109), Vector2(73, 27), "DETACH", _detach_lab.bind(str(lab.get("scene", ""))), color)
	if not bool(record.get("valid", false)):
		open_button.disabled = true
		detach_button.disabled = true
		open_button.text = "LOCKED"
		panel.tooltip_text = ", ".join(record.get("errors", []))
	cards.append({"node": panel, "category": str(lab.get("category", "core"))})


func _apply_filter(filter_name: String) -> void:
	selected_filter = filter_name
	var visible_index := 0
	for card in cards:
		var visible := filter_name == "all" or str(card.get("category", "")) == filter_name
		var node: Control = card.get("node")
		node.visible = visible
		if visible:
			var column := visible_index % 3
			var row := visible_index / 3
			node.position = Vector2(column * 406, row * 160)
			visible_index += 1
	card_surface.custom_minimum_size = Vector2(1214, max(510, int(ceil(visible_index / 3.0)) * 160))
	count_label.text = "%d / %d LABS" % [visible_index, LABS.size()]
	for key in filter_buttons:
		var button: Button = filter_buttons[key]
		button.button_pressed = str(key) == filter_name


func _open_lab(scene_path: String) -> void:
	if ResourceLoader.exists(scene_path):
		get_tree().change_scene_to_file(scene_path)


func _detach_lab(scene_path: String) -> void:
	if ResourceLoader.exists(scene_path):
		var result := OS.create_instance(PackedStringArray([scene_path]))
		if result < 0:
			status_label.text = "DETACHED LAUNCH FAILED // " + scene_path
			status_label.modulate = RED


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == KEY_ESCAPE:
		get_tree().quit()


func _panel_button_style(color: Color, filled: bool) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = Color(color, 0.17 if filled else 0.045)
	style.border_color = Color(color, 0.85 if filled else 0.40)
	style.set_border_width_all(1)
	style.corner_radius_top_left = 2
	style.corner_radius_top_right = 2
	style.corner_radius_bottom_left = 2
	style.corner_radius_bottom_right = 2
	return style


func _label(parent: Node, position: Vector2, size: Vector2, value: String, color := TEXT, font_size := 10) -> Label:
	var result := Label.new()
	result.position = position
	result.size = size
	result.text = value
	result.add_theme_font_size_override("font_size", font_size)
	result.add_theme_color_override("font_color", color)
	parent.add_child(result)
	return result


func _button(parent: Node, position: Vector2, size: Vector2, value: String, callback: Callable, color := CYAN) -> Button:
	var result := Button.new()
	result.position = position
	result.size = size
	result.text = value
	result.toggle_mode = parent is CanvasLayer
	result.focus_mode = Control.FOCUS_NONE
	result.add_theme_font_size_override("font_size", 8)
	result.add_theme_color_override("font_color", color)
	result.add_theme_color_override("font_hover_color", TEXT)
	result.add_theme_stylebox_override("normal", _panel_button_style(color, false))
	result.add_theme_stylebox_override("hover", _panel_button_style(color, true))
	result.add_theme_stylebox_override("pressed", _panel_button_style(color, true))
	result.add_theme_stylebox_override("disabled", _panel_button_style(MUTED, false))
	result.pressed.connect(callback)
	parent.add_child(result)
	return result


func _category_color(category: String) -> Color:
	match category:
		"neural": return CYAN
		"motion": return PINK
		"organism": return LIME
		"ecology": return ORANGE
		"map": return VIOLET
		_: return Color("#8fc7e5")


func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		return {}
	var handle := FileAccess.open(path, FileAccess.READ)
	if handle == null:
		return {}
	var value = JSON.parse_string(handle.get_as_text())
	return value if value is Dictionary else {}


func _nested_value(value: Dictionary, path: String):
	var current: Variant = value
	for part in path.split("."):
		if not current is Dictionary:
			return null
		current = current.get(part, null)
	return current


func _is_sha256(value: String) -> bool:
	if value.length() != 64:
		return false
	for character in value:
		if character not in "0123456789abcdef":
			return false
	return true


func _sha256_file(path: String) -> String:
	if not FileAccess.file_exists(path):
		return "missing"
	var context := HashingContext.new()
	if context.start(HashingContext.HASH_SHA256) != OK:
		return "hash-error"
	context.update(FileAccess.get_file_as_bytes(path))
	return context.finish().hex_encode()


func _run_headless_smoke() -> void:
	var errors := inventory_errors.duplicate()
	var main_scene := str(ProjectSettings.get_setting("application/run/main_scene", ""))
	if main_scene != "res://Arena.tscn":
		errors.append("project main scene is not Arena")
	var valid_lab_count := 0
	for record in inventory:
		if bool(record.get("valid", false)):
			valid_lab_count += 1
	var report := {
		"format": HUB_FORMAT,
		"passed": errors.is_empty(),
		"main_scene": main_scene,
		"lab_count": LABS.size(),
		"valid_lab_count": valid_lab_count,
		"catalog_count": LABS.size() - 1,
		"python_runtime_required": false,
		"filters": FILTERS,
		"inventory": inventory,
		"errors": errors,
	}
	var report_path := ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--forge-hub-report="):
			report_path = argument.trim_prefix("--forge-hub-report=")
	if not report_path.is_empty():
		var absolute_path := report_path if report_path.is_absolute_path() else ProjectSettings.globalize_path(report_path)
		DirAccess.make_dir_recursive_absolute(absolute_path.get_base_dir())
		var handle := FileAccess.open(absolute_path, FileAccess.WRITE)
		if handle == null:
			errors.append("report write")
		else:
			handle.store_string(JSON.stringify(report, "  ", false))
			handle.close()
	if errors.is_empty():
		print("FORGE_HUB_SMOKE_OK labs=%d catalogs=%d main=Arena.tscn" % [LABS.size(), LABS.size() - 1])
	else:
		push_error("FORGE_HUB_SMOKE_FAILED // " + ", ".join(errors))
	get_tree().quit(0 if errors.is_empty() else 1)
