extends Node2D

const INDEX_PATH := "res://generated/v2/asset_index.json"
const PANEL := Color("#08101f")
const PANEL_DEEP := Color("#030711")
const RULE := Color("#233858")
const CYAN := Color("#38ecff")
const MAGENTA := Color("#ff3fb4")
const LIME := Color("#baff57")
const TEXT := Color("#e8f6ff")
const MUTED := Color("#7f94ad")
const WARN := Color("#ffb84d")

var asset_index: Dictionary = {}
var motion_data: Dictionary = {}
var map_data: Dictionary = {}
var motion_clips: Dictionary = {}
var family_atlases: Dictionary = {}
var map_entries: Dictionary = {}
var map_atlases: Dictionary = {}

var family_index := 0
var motion_index := 0
var facing_index := 0
var map_theme_index := 0
var map_layer_index := 0
var motion_frame := 0
var map_frame := 0
var playing := true
var motion_accumulator := 0.0
var map_accumulator := 0.0
var motion_atlas_texture: Texture2D
var map_atlas_texture: Texture2D

var family_names: Array = []
var motion_names: Array = []
var facing_names: Array = []
var theme_names: Array = []
var layer_names: Array = []

var status_label: Label
var family_label: Label
var motion_label: Label
var facing_label: Label
var motion_frame_label: Label
var motion_meta_label: Label
var motion_hash_label: Label
var theme_label: Label
var layer_label: Label
var map_frame_label: Label
var map_meta_label: Label
var map_hash_label: Label
var help_label: Label
var motion_rect: TextureRect
var map_rect: TextureRect
var pause_button: Button


func _ready() -> void:
	get_viewport().set_embedding_subwindows(false)
	_build_interface()
	asset_index = _load_json(INDEX_PATH)
	if asset_index.is_empty():
		_set_unavailable("ASSET INDEX MISSING // RUN python -m forge.forge_lab_sync")
		await _run_smoke_if_requested()
		return
	_load_asset_registry()
	_connect_controls()
	_refresh_all()
	await _run_smoke_if_requested()


func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		return {}
	var handle := FileAccess.open(path, FileAccess.READ)
	if handle == null:
		return {}
	var value = JSON.parse_string(handle.get_as_text())
	return value if value is Dictionary else {}


func _load_asset_registry() -> void:
	motion_data = asset_index.get("motion", {})
	map_data = asset_index.get("maps", {})
	family_names = motion_data.get("families", [])
	motion_names = motion_data.get("motions", [])
	facing_names = motion_data.get("facings", [])
	theme_names = map_data.get("themes", [])
	layer_names = map_data.get("layers", [])
	for atlas_entry in motion_data.get("atlases", []):
		family_atlases[str(atlas_entry.get("family", ""))] = atlas_entry
	for clip in motion_data.get("clips", []):
		motion_clips[_clip_key(clip)] = clip
	for entry in map_data.get("maps", []):
		map_entries[str(entry.get("theme", ""))] = entry
	status_label.text = "BANK ONLINE // %d CLIPS // %d MAPS" % [motion_clips.size(), map_entries.size()]
	status_label.modulate = LIME if not family_names.is_empty() and not theme_names.is_empty() else WARN


func _clip_key(entry: Dictionary) -> String:
	return "%s|%s|%s" % [entry.get("family", ""), entry.get("motion", ""), entry.get("facing", "")]


func _selected_family() -> String:
	return str(family_names[family_index]) if not family_names.is_empty() else "unavailable"


func _selected_motion() -> String:
	return str(motion_names[motion_index]) if not motion_names.is_empty() else "unavailable"


func _selected_facing() -> String:
	return str(facing_names[facing_index]) if not facing_names.is_empty() else "unavailable"


func _selected_theme() -> String:
	return str(theme_names[map_theme_index]) if not theme_names.is_empty() else "unavailable"


func _selected_layer() -> String:
	return str(layer_names[map_layer_index]) if not layer_names.is_empty() else "unavailable"


func _selected_clip() -> Dictionary:
	return motion_clips.get(
		"%s|%s|%s" % [_selected_family(), _selected_motion(), _selected_facing()],
		{}
	)


func _selected_map() -> Dictionary:
	return map_entries.get(_selected_theme(), {})


func _selected_map_layer() -> Dictionary:
	var entry := _selected_map()
	for layer in entry.get("layers", []):
		if str(layer.get("name", "")) == _selected_layer():
			return layer
	return {}


func _build_interface() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)

	var title := Label.new()
	title.position = Vector2(34, 22)
	title.text = "NULLVECTOR // NEURAL FORGE LAB"
	title.add_theme_font_size_override("font_size", 25)
	title.add_theme_color_override("font_color", TEXT)
	layer.add_child(title)

	var subtitle := Label.new()
	subtitle.position = Vector2(36, 55)
	subtitle.text = "GRAPH-RIGGED MORPHOLOGY BANK  x  SEMANTIC MAP ART FORGE"
	subtitle.add_theme_font_size_override("font_size", 10)
	subtitle.add_theme_color_override("font_color", CYAN)
	layer.add_child(subtitle)

	status_label = Label.new()
	status_label.position = Vector2(840, 30)
	status_label.size = Vector2(406, 24)
	status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	status_label.text = "LOADING BANK"
	status_label.add_theme_font_size_override("font_size", 11)
	status_label.add_theme_color_override("font_color", LIME)
	layer.add_child(status_label)

	_build_motion_panel(layer)
	_build_map_panel(layer)

	help_label = Label.new()
	help_label.position = Vector2(32, 676)
	help_label.size = Vector2(1216, 28)
	help_label.text = "Q/E FAMILY   W/S MOTION   A/D FACING   LEFT/RIGHT FRAME   SPACE PLAY   R/F THEME   T/G LAYER   ,/. MAP FRAME"
	help_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	help_label.add_theme_font_size_override("font_size", 10)
	help_label.add_theme_color_override("font_color", MUTED)
	layer.add_child(help_label)


func _panel_rect(parent: Node, rect: Rect2) -> Panel:
	var panel := Panel.new()
	panel.position = rect.position
	panel.size = rect.size
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.025, 0.05, 0.1, 0.94)
	style.border_color = RULE
	style.set_border_width_all(1)
	style.corner_radius_top_left = 4
	style.corner_radius_top_right = 4
	style.corner_radius_bottom_left = 4
	style.corner_radius_bottom_right = 4
	panel.add_theme_stylebox_override("panel", style)
	parent.add_child(panel)
	return panel


func _make_label(parent: Node, position: Vector2, size: Vector2, text: String, color := TEXT, font_size := 11) -> Label:
	var label := Label.new()
	label.position = position
	label.size = size
	label.text = text
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	parent.add_child(label)
	return label


func _make_button(parent: Node, position: Vector2, size: Vector2, text: String, callable: Callable) -> Button:
	var button := Button.new()
	button.position = position
	button.size = size
	button.text = text
	button.focus_mode = Control.FOCUS_NONE
	button.add_theme_font_size_override("font_size", 10)
	button.add_theme_color_override("font_color", TEXT)
	button.add_theme_color_override("font_hover_color", CYAN)
	button.pressed.connect(callable)
	parent.add_child(button)
	return button


func _build_motion_panel(layer: CanvasLayer) -> void:
	var panel := _panel_rect(layer, Rect2(26, 91, 578, 562))
	_make_label(panel, Vector2(18, 12), Vector2(530, 24), "01 // MORPHOLOGY MOTION", CYAN, 13)
	motion_rect = TextureRect.new()
	motion_rect.position = Vector2(25, 48)
	motion_rect.size = Vector2(336, 336)
	motion_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	motion_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	motion_rect.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	motion_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	panel.add_child(motion_rect)

	family_label = _make_label(panel, Vector2(383, 54), Vector2(175, 34), "FAMILY", MAGENTA, 15)
	motion_label = _make_label(panel, Vector2(383, 106), Vector2(175, 34), "MOTION", TEXT, 14)
	facing_label = _make_label(panel, Vector2(383, 158), Vector2(175, 34), "FACING", CYAN, 13)
	motion_frame_label = _make_label(panel, Vector2(383, 210), Vector2(175, 26), "FRAME", MUTED, 11)
	pause_button = _make_button(panel, Vector2(383, 248), Vector2(164, 34), "PAUSE", _toggle_play)

	_make_button(panel, Vector2(25, 397), Vector2(38, 30), "<", func(): _step_family(-1))
	_make_button(panel, Vector2(67, 397), Vector2(38, 30), ">", func(): _step_family(1))
	_make_button(panel, Vector2(117, 397), Vector2(38, 30), "M-", func(): _step_motion(-1))
	_make_button(panel, Vector2(159, 397), Vector2(38, 30), "M+", func(): _step_motion(1))
	_make_button(panel, Vector2(209, 397), Vector2(38, 30), "F-", func(): _step_facing(-1))
	_make_button(panel, Vector2(251, 397), Vector2(38, 30), "F+", func(): _step_facing(1))
	_make_button(panel, Vector2(301, 397), Vector2(38, 30), "-1", func(): _step_motion_frame(-1))
	_make_button(panel, Vector2(343, 397), Vector2(38, 30), "+1", func(): _step_motion_frame(1))

	motion_meta_label = _make_label(panel, Vector2(25, 441), Vector2(522, 55), "", MUTED, 10)
	motion_meta_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	motion_hash_label = _make_label(panel, Vector2(25, 506), Vector2(522, 38), "", CYAN, 9)
	motion_hash_label.autowrap_mode = TextServer.AUTOWRAP_ARBITRARY


func _build_map_panel(layer: CanvasLayer) -> void:
	var panel := _panel_rect(layer, Rect2(622, 91, 632, 562))
	_make_label(panel, Vector2(18, 12), Vector2(590, 24), "02 // PROCEDURAL MAP SEMANTICS", MAGENTA, 13)
	map_rect = TextureRect.new()
	map_rect.position = Vector2(24, 48)
	map_rect.size = Vector2(430, 430)
	map_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	map_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	map_rect.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	map_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	panel.add_child(map_rect)

	theme_label = _make_label(panel, Vector2(474, 55), Vector2(143, 34), "THEME", MAGENTA, 14)
	layer_label = _make_label(panel, Vector2(474, 107), Vector2(143, 48), "LAYER", CYAN, 12)
	map_frame_label = _make_label(panel, Vector2(474, 164), Vector2(143, 26), "FRAME", MUTED, 10)
	_make_button(panel, Vector2(474, 204), Vector2(63, 30), "THEME-", func(): _step_theme(-1))
	_make_button(panel, Vector2(544, 204), Vector2(63, 30), "THEME+", func(): _step_theme(1))
	_make_button(panel, Vector2(474, 242), Vector2(63, 30), "LAYER-", func(): _step_layer(-1))
	_make_button(panel, Vector2(544, 242), Vector2(63, 30), "LAYER+", func(): _step_layer(1))
	_make_button(panel, Vector2(474, 280), Vector2(63, 30), "FRAME-", func(): _step_map_frame(-1))
	_make_button(panel, Vector2(544, 280), Vector2(63, 30), "FRAME+", func(): _step_map_frame(1))

	map_meta_label = _make_label(panel, Vector2(474, 330), Vector2(143, 126), "", MUTED, 9)
	map_meta_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	map_hash_label = _make_label(panel, Vector2(24, 492), Vector2(583, 48), "", CYAN, 9)
	map_hash_label.autowrap_mode = TextServer.AUTOWRAP_ARBITRARY


func _connect_controls() -> void:
	# The scene is built in code so asset contracts, UI state, and the smoke path
	# are exercised through the same runtime path.
	pass


func _set_unavailable(message: String) -> void:
	status_label.text = message
	status_label.modulate = WARN
	motion_meta_label.text = "Motion bank unavailable. The scene remains safe to open while generation is in progress."
	map_meta_label.text = "Map bank unavailable."


func _load_texture(path: String) -> Texture2D:
	if path.is_empty() or not ResourceLoader.exists(path):
		return null
	var resource = load(path)
	return resource if resource is Texture2D else null


func _refresh_all() -> void:
	_refresh_motion(true)
	_refresh_map(true)
	queue_redraw()


func _refresh_motion(reload_atlas := false) -> void:
	var clip := _selected_clip()
	var atlas_entry: Dictionary = family_atlases.get(_selected_family(), {})
	if clip.is_empty() or atlas_entry.is_empty():
		motion_rect.texture = null
		motion_meta_label.text = "No clip for this family / motion / facing."
		return
	if reload_atlas or motion_atlas_texture == null:
		motion_atlas_texture = _load_texture("res://generated/v2/%s" % atlas_entry.get("atlas", ""))
	motion_frame = posmod(motion_frame, int(clip.get("frame_count", 1)))
	var cell := int(clip.get("start_cell", 0)) + motion_frame
	var region := AtlasTexture.new()
	region.atlas = motion_atlas_texture
	var cell_size := int(clip.get("cell_size", 48))
	var columns := int(clip.get("atlas_columns", 16))
	region.region = Rect2((cell % columns) * cell_size, (cell / columns) * cell_size, cell_size, cell_size)
	motion_rect.texture = region
	family_label.text = _selected_family().to_upper()
	motion_label.text = _selected_motion().to_upper().replace("_", " ")
	facing_label.text = _selected_facing().to_upper()
	motion_frame_label.text = "FRAME %02d / %02d   %d FPS" % [motion_frame + 1, clip.get("frame_count", 1), clip.get("fps", 0)]
	var metrics: Dictionary = clip.get("metrics", {})
	motion_meta_label.text = "SOURCE %s\nLOOP %s  UNIQUE %s  DELTA-PIX %s\nROOT %s  FEET L%s R%s" % [
		atlas_entry.get("source_id", "unknown"),
		"YES" if clip.get("loop", false) else "NO",
		metrics.get("unique_semantic_frames", "?"),
		metrics.get("max_changed_pixel_fraction", "?"),
		metrics.get("root_span", []),
		metrics.get("left_foot_span", []),
		metrics.get("right_foot_span", []),
	]
	motion_hash_label.text = "CLIP %s\nATLAS %s" % [
		str(clip.get("clip_sha256", "missing")),
		str(atlas_entry.get("atlas_sha256", "missing")),
	]


func _refresh_map(reload_atlas := false) -> void:
	var entry := _selected_map()
	var map_layer := _selected_map_layer()
	if entry.is_empty() or map_layer.is_empty():
		map_rect.texture = null
		map_meta_label.text = "No map layer for this selection."
		return
	if reload_atlas or map_atlas_texture == null:
		map_atlas_texture = _load_texture("res://generated/v2/%s" % entry.get("atlas", ""))
	map_frame = posmod(map_frame, int(map_layer.get("frame_count", 1)))
	var cell := int(map_layer.get("start_cell", 0)) + map_frame
	var region := AtlasTexture.new()
	region.atlas = map_atlas_texture
	var cell_size := int(entry.get("cell_size", 384))
	var columns := int(entry.get("columns", 4))
	region.region = Rect2((cell % columns) * cell_size, (cell / columns) * cell_size, cell_size, cell_size)
	map_rect.texture = region
	theme_label.text = _selected_theme().to_upper()
	layer_label.text = _selected_layer().to_upper().replace("_", " ")
	map_frame_label.text = "FRAME %02d / %02d" % [map_frame + 1, map_layer.get("frame_count", 1)]
	var stats: Dictionary = entry.get("statistics", {})
	map_meta_label.text = "SEED %s\nINST %s  PROP %s\nHAZARD %s\nCOLLISION %s\nEMISSIVE %s" % [
		entry.get("seed", "?"),
		stats.get("instance_count", "?"),
		stats.get("prop_count", "?"),
		stats.get("animated_hazard_cells", "?"),
		stats.get("collision_cells", "?"),
		stats.get("emissive_pixels", "?"),
	]
	var renderer: Dictionary = entry.get("renderer", {})
	map_hash_label.text = "MAP %s  //  RENDERER %s\nATLAS %s" % [
		str(entry.get("source_semantic_sha256", "missing")),
		str(renderer.get("source_sha256", "missing")),
		str(entry.get("atlas_sha256", "missing")),
	]


func _step_family(delta: int) -> void:
	if family_names.is_empty(): return
	family_index = posmod(family_index + delta, family_names.size())
	motion_frame = 0
	motion_accumulator = 0.0
	_refresh_motion(true)


func _step_motion(delta: int) -> void:
	if motion_names.is_empty(): return
	motion_index = posmod(motion_index + delta, motion_names.size())
	motion_frame = 0
	motion_accumulator = 0.0
	_refresh_motion()


func _step_facing(delta: int) -> void:
	if facing_names.is_empty(): return
	facing_index = posmod(facing_index + delta, facing_names.size())
	motion_frame = 0
	motion_accumulator = 0.0
	_refresh_motion()


func _step_motion_frame(delta: int) -> void:
	var clip := _selected_clip()
	if clip.is_empty(): return
	motion_frame = posmod(motion_frame + delta, int(clip.get("frame_count", 1)))
	motion_accumulator = 0.0
	_refresh_motion()


func _step_theme(delta: int) -> void:
	if theme_names.is_empty(): return
	map_theme_index = posmod(map_theme_index + delta, theme_names.size())
	map_frame = 0
	map_accumulator = 0.0
	_refresh_map(true)


func _step_layer(delta: int) -> void:
	if layer_names.is_empty(): return
	map_layer_index = posmod(map_layer_index + delta, layer_names.size())
	map_frame = 0
	map_accumulator = 0.0
	_refresh_map()


func _step_map_frame(delta: int) -> void:
	var map_layer := _selected_map_layer()
	if map_layer.is_empty(): return
	map_frame = posmod(map_frame + delta, int(map_layer.get("frame_count", 1)))
	map_accumulator = 0.0
	_refresh_map()


func _toggle_play() -> void:
	playing = not playing
	pause_button.text = "PAUSE" if playing else "PLAY"


func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey) or not event.pressed or event.echo:
		return
	match (event as InputEventKey).keycode:
		KEY_Q: _step_family(-1)
		KEY_E: _step_family(1)
		KEY_W: _step_motion(-1)
		KEY_S: _step_motion(1)
		KEY_A: _step_facing(-1)
		KEY_D: _step_facing(1)
		KEY_LEFT: _step_motion_frame(-1)
		KEY_RIGHT: _step_motion_frame(1)
		KEY_SPACE: _toggle_play()
		KEY_R: _step_theme(-1)
		KEY_F: _step_theme(1)
		KEY_T: _step_layer(-1)
		KEY_G: _step_layer(1)
		KEY_COMMA: _step_map_frame(-1)
		KEY_PERIOD: _step_map_frame(1)


func _process(delta: float) -> void:
	if not playing:
		return
	var clip := _selected_clip()
	if not clip.is_empty():
		motion_accumulator += delta
		var interval := 1.0 / maxf(1.0, float(clip.get("fps", 1)))
		if motion_accumulator >= interval:
			motion_accumulator = fmod(motion_accumulator, interval)
			var next := motion_frame + 1
			if next >= int(clip.get("frame_count", 1)):
				next = 0 if clip.get("loop", false) else motion_frame
			if next != motion_frame:
				motion_frame = next
				_refresh_motion()
	var map_layer := _selected_map_layer()
	var map_fps := float(map_layer.get("fps", 0.0))
	if not map_layer.is_empty() and map_fps > 0.0:
		map_accumulator += delta
		var map_interval := 1.0 / map_fps
		if map_accumulator >= map_interval:
			map_accumulator = fmod(map_accumulator, map_interval)
			map_frame = posmod(map_frame + 1, int(map_layer.get("frame_count", 1)))
			_refresh_map()


func _draw() -> void:
	draw_rect(Rect2(0, 0, 1280, 720), PANEL_DEEP)
	for x in range(0, 1281, 32):
		var alpha := 0.075 if x % 128 == 0 else 0.025
		draw_line(Vector2(x, 80), Vector2(x, 665), Color(0.15, 0.55, 0.75, alpha), 1.0)
	for y in range(80, 666, 32):
		var alpha := 0.075 if y % 128 == 16 else 0.025
		draw_line(Vector2(0, y), Vector2(1280, y), Color(0.15, 0.55, 0.75, alpha), 1.0)
	draw_line(Vector2(28, 78), Vector2(1252, 78), RULE, 1.0)


func _asset_path(relative: Variant) -> String:
	return "res://generated/v2/%s" % str(relative)


func _check_file_hash(path: String, expected: Variant, label: String, errors: Array[String]) -> void:
	if not FileAccess.file_exists(path):
		errors.append("missing %s" % label)
		return
	if FileAccess.get_sha256(path) != str(expected):
		errors.append("hash %s" % label)


func _smoke_option(prefix: String) -> String:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with(prefix):
			return argument.trim_prefix(prefix)
	return ""


func _write_smoke_report(path: String, report: Dictionary, errors: Array[String]) -> void:
	if path.is_empty():
		return
	var handle := FileAccess.open(path, FileAccess.WRITE)
	if handle == null:
		errors.append("report write %s" % path)
		return
	handle.store_string(JSON.stringify(report, "  ") + "\n")
	handle.close()


func _run_smoke_if_requested() -> void:
	if "--forge-lab-smoke" not in OS.get_cmdline_user_args():
		return
	var errors: Array[String] = []
	var atlas_hashes_checked := 0
	var source_hashes_checked := 0
	var motion_regions_checked := 0
	var map_regions_checked := 0
	var motion_frames_checked := 0
	var map_frames_checked := 0
	var expected_families := ["humanoid", "animalian", "plantlike", "anomaly", "machine"]
	var expected_motions := ["idle_breathe", "idle_wiggle", "locomote", "joy", "anger", "fear", "confused", "sleep", "taunt", "attack", "cast", "hit", "death"]
	var expected_facings := ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
	var expected_themes := ["arena", "rooms", "caves", "archipelago", "garden", "anomaly"]
	var expected_layers := ["composite", "base_color", "emissive", "collision", "occlusion", "autotile", "elevation_edges", "objects", "hazard"]

	if asset_index.get("format", "") != "nullvector-forge-lab-assets-v1": errors.append("index format")
	if asset_index.get("engine", "") != "Godot 4.3": errors.append("engine contract")
	if asset_index.get("pixel_filter", "") != "nearest": errors.append("index pixel filter")
	if bool(asset_index.get("python_runtime_required", true)): errors.append("python runtime contract")
	if asset_index.get("runtime_asset_extensions", []) != [".json", ".png"]: errors.append("runtime extensions")
	if not asset_index.get("errors", []).is_empty(): errors.append("index subsystem errors")
	if family_names != expected_families: errors.append("family contract")
	if motion_names != expected_motions: errors.append("motion contract")
	if facing_names != expected_facings: errors.append("facing contract")
	if theme_names != expected_themes: errors.append("theme contract")
	if layer_names != expected_layers: errors.append("layer contract")
	if motion_clips.size() != 520: errors.append("clip count")
	if map_entries.size() != 6: errors.append("map entry count")
	if family_atlases.size() != 5: errors.append("motion atlas count")
	if motion_rect.texture_filter != CanvasItem.TEXTURE_FILTER_NEAREST: errors.append("motion rect filter")
	if map_rect.texture_filter != CanvasItem.TEXTURE_FILTER_NEAREST: errors.append("map rect filter")
	if int(ProjectSettings.get_setting("rendering/textures/canvas_textures/default_texture_filter", -1)) != 0:
		errors.append("project texture filter")
	if bool(ProjectSettings.get_setting("rendering/textures/default_filters/use_nearest_mipmap_filter", true)):
		errors.append("project mip filter")

	var source_manifest_path := _asset_path(motion_data.get("source_manifest", ""))
	_check_file_hash(source_manifest_path, motion_data.get("source_manifest_sha256", ""), "motion source", errors)
	source_hashes_checked += 1
	for family_position in range(family_names.size()):
		family_index = family_position
		var family := str(family_names[family_position])
		var atlas_entry: Dictionary = family_atlases.get(family, {})
		if atlas_entry.is_empty():
			errors.append("missing motion atlas %s" % family)
			continue
		var atlas_path := _asset_path(atlas_entry.get("atlas", ""))
		_check_file_hash(atlas_path, atlas_entry.get("atlas_sha256", ""), "motion atlas %s" % family, errors)
		atlas_hashes_checked += 1
		var texture := _load_texture(atlas_path)
		if texture == null:
			errors.append("load motion atlas %s" % family)
			continue
		var expected_size: Array = atlas_entry.get("atlas_size", [])
		if expected_size.size() != 2 or texture.get_width() != int(expected_size[0]) or texture.get_height() != int(expected_size[1]):
			errors.append("size motion atlas %s" % family)
		var columns := int(atlas_entry.get("columns", 0))
		var rows := int(atlas_entry.get("rows", 0))
		var cell_size := int(atlas_entry.get("cell_size", 0))
		if columns <= 0 or rows <= 0 or cell_size != 48:
			errors.append("grid motion atlas %s" % family)
		for motion_position in range(motion_names.size()):
			motion_index = motion_position
			for facing_position in range(facing_names.size()):
				facing_index = facing_position
				motion_frame = 0
				var clip := _selected_clip()
				var label := "%s/%s/%s" % [family, _selected_motion(), _selected_facing()]
				if clip.is_empty():
					errors.append("missing clip %s" % label)
					continue
				var frame_count := int(clip.get("frame_count", 0))
				var start_cell := int(clip.get("start_cell", -1))
				if frame_count <= 0 or clip.get("frame_sha256", []).size() != frame_count:
					errors.append("frame contract %s" % label)
				if int(clip.get("atlas_columns", -1)) != columns or int(clip.get("cell_size", -1)) != cell_size:
					errors.append("atlas contract %s" % label)
				for local_frame in range(frame_count):
					var cell := start_cell + local_frame
					var x := (cell % columns) * cell_size
					var y := (cell / columns) * cell_size
					if cell < 0 or cell >= columns * rows or x < 0 or y < 0 or x + cell_size > texture.get_width() or y + cell_size > texture.get_height():
						errors.append("region %s/%d" % [label, local_frame])
						break
					motion_frames_checked += 1
				motion_regions_checked += 1
				_refresh_motion(facing_position == 0 and motion_position == 0)
				if motion_rect.texture == null or motion_rect.texture.get_width() != cell_size or motion_rect.texture.get_height() != cell_size:
					errors.append("ui motion region %s" % label)

	for theme_position in range(theme_names.size()):
		map_theme_index = theme_position
		var theme := str(theme_names[theme_position])
		var entry: Dictionary = map_entries.get(theme, {})
		if entry.is_empty():
			errors.append("missing map %s" % theme)
			continue
		var source_path := _asset_path(entry.get("source_manifest", ""))
		_check_file_hash(source_path, entry.get("source_manifest_sha256", ""), "map source %s" % theme, errors)
		source_hashes_checked += 1
		var atlas_path := _asset_path(entry.get("atlas", ""))
		_check_file_hash(atlas_path, entry.get("atlas_sha256", ""), "map atlas %s" % theme, errors)
		atlas_hashes_checked += 1
		var texture := _load_texture(atlas_path)
		if texture == null:
			errors.append("load map atlas %s" % theme)
			continue
		var expected_size: Array = entry.get("atlas_size", [])
		if expected_size.size() != 2 or texture.get_width() != int(expected_size[0]) or texture.get_height() != int(expected_size[1]):
			errors.append("size map atlas %s" % theme)
		var columns := int(entry.get("columns", 0))
		var rows := int(entry.get("rows", 0))
		var cell_size := int(entry.get("cell_size", 0))
		var layers: Array = entry.get("layers", [])
		if layers.size() != layer_names.size(): errors.append("map layers %s" % theme)
		for layer_position in range(layer_names.size()):
			map_layer_index = layer_position
			map_frame = 0
			var map_layer := _selected_map_layer()
			var label := "%s/%s" % [theme, _selected_layer()]
			if map_layer.is_empty():
				errors.append("missing map layer %s" % label)
				continue
			var frame_count := int(map_layer.get("frame_count", 0))
			var start_cell := int(map_layer.get("start_cell", -1))
			for local_frame in range(frame_count):
				var cell := start_cell + local_frame
				var x := (cell % columns) * cell_size
				var y := (cell / columns) * cell_size
				if cell < 0 or cell >= columns * rows or x < 0 or y < 0 or x + cell_size > texture.get_width() or y + cell_size > texture.get_height():
					errors.append("region %s/%d" % [label, local_frame])
					break
				map_frames_checked += 1
			map_regions_checked += 1
			_refresh_map(layer_position == 0)
			if map_rect.texture == null or map_rect.texture.get_width() != cell_size or map_rect.texture.get_height() != cell_size:
				errors.append("ui map region %s" % label)

	# Leave the capture on an animated, readable state after exhausting every
	# selector path above.
	family_index = expected_families.find("humanoid")
	motion_index = expected_motions.find("locomote")
	facing_index = expected_facings.find("southeast")
	motion_frame = 2
	map_theme_index = expected_themes.find("anomaly")
	map_layer_index = expected_layers.find("composite")
	map_frame = 0
	_refresh_all()

	var report := {
		"format": "nullvector-forge-lab-smoke-v1",
		"passed": errors.is_empty(),
		"engine": Engine.get_version_info().get("string", "unknown"),
		"display_server": DisplayServer.get_name(),
		"renderer": str(ProjectSettings.get_setting("rendering/renderer/rendering_method", "unknown")),
		"scene": scene_file_path,
		"index": INDEX_PATH,
		"nearest_filtering": true,
		"python_runtime_required": false,
		"coverage": {
			"families": family_names.size(),
			"motions": motion_names.size(),
			"facings": facing_names.size(),
			"motion_clips": motion_regions_checked,
			"motion_frames": motion_frames_checked,
			"map_themes": theme_names.size(),
			"map_layers_per_theme": layer_names.size(),
			"map_regions": map_regions_checked,
			"map_frames": map_frames_checked,
			"atlas_hashes": atlas_hashes_checked,
			"source_manifest_hashes": source_hashes_checked,
		},
		"errors": errors,
	}
	_write_smoke_report(_smoke_option("--forge-lab-report="), report, errors)
	if errors.is_empty():
		print("FORGE_LAB_SMOKE_OK families=5 motions=13 facings=8 clips=520 motion_frames=%d maps=6 layers=9 map_regions=%d map_frames=%d hashes=%d" % [motion_frames_checked, map_regions_checked, map_frames_checked, atlas_hashes_checked + source_hashes_checked])
		get_tree().quit(0)
	else:
		push_error("FORGE_LAB_SMOKE_FAILED " + ", ".join(errors))
		get_tree().quit(1)
