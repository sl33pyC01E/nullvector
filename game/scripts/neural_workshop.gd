extends Node2D

const INDEX_PATH := "res://generated/neural_workshop/v1/asset_index.json"
const INDEX_FORMAT := "nullvector-neural-workshop-assets-v1"
const STATIC_LAYERS := ["base", "outline", "emission_core", "aura", "bloom_r1", "bloom_r2", "composite"]
const EXPECTED_FAMILIES := ["humanoid", "animalian", "plantlike", "anomaly", "machine"]
const EXPECTED_ROLES := ["striker", "defender", "scout", "controller", "support", "artillery", "harvester", "disruptor"]
const EXPECTED_MOTIONS := ["idle_breathe", "idle_wiggle", "locomote", "joy", "anger", "fear", "confused", "sleep", "taunt", "attack", "cast", "hit", "death"]
const EXPECTED_FACINGS := ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]
const EXPECTED_THEMES := ["arena", "rooms", "caves", "archipelago", "garden", "anomaly"]
const EXPECTED_MAP_LAYERS := ["composite", "base_color", "emissive", "collision", "occlusion", "autotile", "elevation_edges", "objects", "hazard", "protected_backbone", "required_clearance", "decoration_forbidden", "walkability", "hazard_semantic", "zones", "nav_cost"]
const EXPECTED_MOTION_REPRESENTATIVES := {
	"humanoid": ["0000_f0_s00_r0_v00", 0],
	"animalian": ["0016_f1_s04_r0_v00", 16],
	"plantlike": ["0032_f2_s08_r0_v00", 32],
	"anomaly": ["0048_f3_s12_r0_v00", 48],
	"machine": ["0064_f4_s16_r0_v00", 64],
}
const SOURCE_IDENTITY_MANIFEST_SEMANTICS := "verbatim-upstream-audit-copy; embedded artifact paths are upstream-root-relative and are not runtime-resolvable"
const ALL_FILTER := "all"
const PANEL_DEEP := Color("#030711")
const PANEL := Color("#071224")
const RULE := Color("#203a5f")
const CYAN := Color("#38ecff")
const MAGENTA := Color("#ff3fb4")
const LIME := Color("#baff57")
const TEXT := Color("#e8f6ff")
const MUTED := Color("#7890aa")
const WARN := Color("#ffb84d")
const ERROR := Color("#ff526d")

var asset_index: Dictionary = {}
var static_data: Dictionary = {}
var motion_data: Dictionary = {}
var map_data: Dictionary = {}
var identities: Array = []
var filtered_identities: Array = []
var static_atlases: Dictionary = {}
var motion_identities: Dictionary = {}
var map_entries: Dictionary = {}
var map_atlas_texture: Texture2D
var static_atlas_texture: Texture2D
var motion_atlas_texture: Texture2D

var selected_identity := 0
var selected_layer := 6
var selected_family_filter := 0
var selected_subtype_filter := 0
var selected_role_filter := 0
var selected_scale := 4
var selected_motion := 0
var selected_facing := 4
var selected_motion_frame := 0
var selected_theme := 0
var selected_map_layer := 0
var selected_map_frame := 0
var playing := true
var motion_accumulator := 0.0
var map_accumulator := 0.0
var startup_errors: Array[String] = []

var status_label: Label
var filter_label: Label
var identity_label: Label
var layer_label: Label
var scale_label: Label
var static_hash_label: Label
var static_rect: TextureRect
var motion_status_label: Label
var motion_label: Label
var facing_label: Label
var motion_frame_label: Label
var motion_hash_label: Label
var motion_rect: TextureRect
var pause_button: Button
var theme_label: Label
var map_layer_label: Label
var map_frame_label: Label
var map_meta_label: Label
var map_hash_label: Label
var map_rect: TextureRect
var help_label: Label


func _ready() -> void:
	get_viewport().set_embedding_subwindows(false)
	_build_interface()
	asset_index = _load_json(INDEX_PATH)
	if asset_index.is_empty():
		_set_unavailable("WORKSHOP INDEX MISSING // RUN python -m forge.neural_workshop_sync")
		await _run_smoke_if_requested()
		return
	_load_registry()
	_apply_filters(true)
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


func _load_registry() -> void:
	if asset_index.get("format", "") != INDEX_FORMAT:
		startup_errors.append("index format")
	if asset_index.get("status", "") != "ready":
		startup_errors.append("index status")
	if asset_index.get("pixel_filter", "") != "nearest":
		startup_errors.append("pixel filter")
	if bool(asset_index.get("python_runtime_required", true)):
		startup_errors.append("runtime Python contract")
	if not asset_index.get("errors", []).is_empty():
		startup_errors.append("source index errors")
	static_data = asset_index.get("static", {})
	motion_data = asset_index.get("motion", {})
	map_data = asset_index.get("maps", {})
	if static_data.get("status", "") != "ready": startup_errors.append("static bank")
	if map_data.get("status", "") != "ready": startup_errors.append("map bank")
	identities = static_data.get("identities", [])
	for atlas in static_data.get("atlases", []):
		static_atlases[str(atlas.get("layer", ""))] = atlas
	if motion_data.get("status", "") == "ready" and bool(motion_data.get("available", false)) and bool(motion_data.get("neural_output", false)):
		for entry in motion_data.get("identities", []):
			motion_identities[str(entry.get("family", ""))] = entry
	for entry in map_data.get("maps", []):
		map_entries[str(entry.get("theme", ""))] = entry
	if startup_errors.is_empty():
		var motion_note := "NEURAL MOTION ONLINE"
		if not _motion_available():
			motion_note = "NEURAL MOTION REJECTED / FAIL-CLOSED" if motion_data.get("status", "") == "rejected" else "NEURAL MOTION STAGED / FAIL-CLOSED"
		status_label.text = "BUNDLE %s  //  80 IDENTITIES  //  6 TOPOLOGY-v2 MAPS  //  %s" % [str(asset_index.get("bundle_id", "missing")).substr(0, 12), motion_note]
		status_label.modulate = LIME if _motion_available() else (ERROR if motion_data.get("status", "") == "rejected" else WARN)
	else:
		_set_unavailable("WORKSHOP CONTRACT REJECTED // " + ", ".join(startup_errors))


func _motion_available() -> bool:
	return motion_data.get("status", "") == "ready" and bool(motion_data.get("available", false)) and bool(motion_data.get("neural_output", false)) and motion_identities.size() == EXPECTED_FAMILIES.size()


func _build_interface() -> void:
	var canvas := CanvasLayer.new()
	add_child(canvas)
	var title := _make_label(canvas, Vector2(28, 16), Vector2(700, 32), "NULLVECTOR // NATIVE NEURAL ASSET WORKSHOP", TEXT, 23)
	title.add_theme_color_override("font_shadow_color", Color(0.0, 0.8, 1.0, 0.22))
	_make_label(canvas, Vector2(30, 48), Vector2(760, 22), "ACTUAL NEURAL IDENTITIES  x  DERIVED LAYERS  x  TOPOLOGY-v2 MAPS", CYAN, 10)
	status_label = _make_label(canvas, Vector2(760, 21), Vector2(492, 36), "LOADING WORKSHOP BUNDLE", LIME, 10)
	status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	_build_identity_panel(canvas)
	_build_motion_panel(canvas)
	_build_map_panel(canvas)
	help_label = _make_label(canvas, Vector2(24, 682), Vector2(1232, 25), "Q/E IDENTITY  1/2/3 FILTERS  Z/X LAYER  V SCALE  W/S MOTION  A/D FACING  LEFT/RIGHT FRAME  SPACE PLAY  R/F THEME  T/G MAP LAYER  ,/. MAP FRAME", MUTED, 9)
	help_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER


func _panel(parent: Node, rect: Rect2) -> Panel:
	var panel := Panel.new()
	panel.position = rect.position
	panel.size = rect.size
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.025, 0.055, 0.11, 0.96)
	style.border_color = RULE
	style.set_border_width_all(1)
	style.corner_radius_top_left = 4
	style.corner_radius_top_right = 4
	style.corner_radius_bottom_left = 4
	style.corner_radius_bottom_right = 4
	panel.add_theme_stylebox_override("panel", style)
	parent.add_child(panel)
	return panel


func _make_label(parent: Node, position: Vector2, size: Vector2, value: String, color := TEXT, font_size := 11) -> Label:
	var label := Label.new()
	label.position = position
	label.size = size
	label.text = value
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	parent.add_child(label)
	return label


func _make_button(parent: Node, position: Vector2, size: Vector2, value: String, callable: Callable) -> Button:
	var button := Button.new()
	button.position = position
	button.size = size
	button.text = value
	button.focus_mode = Control.FOCUS_NONE
	button.add_theme_font_size_override("font_size", 9)
	button.add_theme_color_override("font_color", TEXT)
	button.add_theme_color_override("font_hover_color", CYAN)
	button.pressed.connect(callable)
	parent.add_child(button)
	return button


func _make_texture_rect(parent: Node, position: Vector2, size: Vector2) -> TextureRect:
	var rect := TextureRect.new()
	rect.position = position
	rect.size = size
	rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	rect.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	parent.add_child(rect)
	return rect


func _build_identity_panel(canvas: CanvasLayer) -> void:
	var panel := _panel(canvas, Rect2(20, 82, 414, 584))
	_make_label(panel, Vector2(16, 10), Vector2(380, 22), "01 // NEURAL IDENTITY + LAYERS", CYAN, 12)
	static_rect = _make_texture_rect(panel, Vector2(16, 39), Vector2(256, 256))
	identity_label = _make_label(panel, Vector2(286, 48), Vector2(112, 126), "IDENTITY", MAGENTA, 11)
	identity_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	layer_label = _make_label(panel, Vector2(286, 181), Vector2(112, 42), "LAYER", CYAN, 10)
	scale_label = _make_label(panel, Vector2(286, 231), Vector2(112, 28), "SCALE", LIME, 10)
	_make_button(panel, Vector2(286, 263), Vector2(51, 27), "LAYER-", func(): _step_layer(-1))
	_make_button(panel, Vector2(343, 263), Vector2(51, 27), "LAYER+", func(): _step_layer(1))
	_make_button(panel, Vector2(16, 307), Vector2(38, 28), "<", func(): _step_identity(-1))
	_make_button(panel, Vector2(59, 307), Vector2(38, 28), ">", func(): _step_identity(1))
	_make_button(panel, Vector2(106, 307), Vector2(86, 28), "FAMILY", func(): _step_family_filter(1))
	_make_button(panel, Vector2(198, 307), Vector2(86, 28), "SUBTYPE", func(): _step_subtype_filter(1))
	_make_button(panel, Vector2(290, 307), Vector2(86, 28), "ROLE", func(): _step_role_filter(1))
	_make_button(panel, Vector2(16, 342), Vector2(108, 28), "CLEAR FILTERS", _clear_filters)
	_make_button(panel, Vector2(130, 342), Vector2(88, 28), "1x / 4x", _toggle_scale)
	filter_label = _make_label(panel, Vector2(16, 383), Vector2(382, 72), "FILTER", MUTED, 9)
	filter_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	static_hash_label = _make_label(panel, Vector2(16, 466), Vector2(382, 101), "HASH", CYAN, 8)
	static_hash_label.autowrap_mode = TextServer.AUTOWRAP_ARBITRARY


func _build_motion_panel(canvas: CanvasLayer) -> void:
	var panel := _panel(canvas, Rect2(447, 82, 392, 584))
	_make_label(panel, Vector2(16, 10), Vector2(360, 22), "02 // DERIVED NEURAL MOTION", MAGENTA, 12)
	motion_rect = _make_texture_rect(panel, Vector2(20, 42), Vector2(256, 256))
	motion_status_label = _make_label(panel, Vector2(288, 48), Vector2(90, 108), "STAGED", WARN, 10)
	motion_status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	motion_label = _make_label(panel, Vector2(288, 166), Vector2(90, 54), "MOTION", TEXT, 9)
	motion_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	facing_label = _make_label(panel, Vector2(288, 224), Vector2(90, 34), "FACING", CYAN, 9)
	motion_frame_label = _make_label(panel, Vector2(288, 264), Vector2(90, 30), "FRAME", MUTED, 9)
	_make_button(panel, Vector2(20, 310), Vector2(52, 28), "M-", func(): _step_motion(-1))
	_make_button(panel, Vector2(78, 310), Vector2(52, 28), "M+", func(): _step_motion(1))
	_make_button(panel, Vector2(136, 310), Vector2(52, 28), "F-", func(): _step_facing(-1))
	_make_button(panel, Vector2(194, 310), Vector2(52, 28), "F+", func(): _step_facing(1))
	_make_button(panel, Vector2(252, 310), Vector2(52, 28), "-1", func(): _step_motion_frame(-1))
	_make_button(panel, Vector2(310, 310), Vector2(52, 28), "+1", func(): _step_motion_frame(1))
	pause_button = _make_button(panel, Vector2(20, 346), Vector2(110, 30), "PAUSE", _toggle_play)
	motion_hash_label = _make_label(panel, Vector2(20, 395), Vector2(352, 170), "Neural motion is staged until the authoritative neural presentation bank passes exact replay.", MUTED, 8)
	motion_hash_label.autowrap_mode = TextServer.AUTOWRAP_ARBITRARY


func _build_map_panel(canvas: CanvasLayer) -> void:
	var panel := _panel(canvas, Rect2(852, 82, 408, 584))
	_make_label(panel, Vector2(16, 10), Vector2(376, 22), "03 // TOPOLOGY-v2 MAP LAYERS", LIME, 12)
	map_rect = _make_texture_rect(panel, Vector2(14, 42), Vector2(310, 310))
	theme_label = _make_label(panel, Vector2(330, 48), Vector2(64, 54), "THEME", MAGENTA, 9)
	theme_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	map_layer_label = _make_label(panel, Vector2(330, 116), Vector2(64, 82), "LAYER", CYAN, 8)
	map_layer_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	map_frame_label = _make_label(panel, Vector2(330, 211), Vector2(64, 34), "FRAME", MUTED, 8)
	_make_button(panel, Vector2(330, 258), Vector2(30, 26), "-", func(): _step_theme(-1))
	_make_button(panel, Vector2(364, 258), Vector2(30, 26), "+", func(): _step_theme(1))
	_make_button(panel, Vector2(14, 363), Vector2(65, 28), "THEME-", func(): _step_theme(-1))
	_make_button(panel, Vector2(84, 363), Vector2(65, 28), "THEME+", func(): _step_theme(1))
	_make_button(panel, Vector2(154, 363), Vector2(65, 28), "LAYER-", func(): _step_map_layer(-1))
	_make_button(panel, Vector2(224, 363), Vector2(65, 28), "LAYER+", func(): _step_map_layer(1))
	_make_button(panel, Vector2(294, 363), Vector2(47, 28), "-1", func(): _step_map_frame(-1))
	_make_button(panel, Vector2(346, 363), Vector2(47, 28), "+1", func(): _step_map_frame(1))
	map_meta_label = _make_label(panel, Vector2(14, 405), Vector2(379, 77), "MAP", MUTED, 8)
	map_meta_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	map_hash_label = _make_label(panel, Vector2(14, 490), Vector2(379, 77), "HASH", CYAN, 8)
	map_hash_label.autowrap_mode = TextServer.AUTOWRAP_ARBITRARY


func _set_unavailable(message: String) -> void:
	status_label.text = message
	status_label.modulate = ERROR
	static_rect.texture = null
	motion_rect.texture = null
	map_rect.texture = null


func _asset_path(relative: Variant) -> String:
	return "res://generated/neural_workshop/v1/%s" % str(relative)


func _load_texture(relative: Variant) -> Texture2D:
	var path := _asset_path(relative)
	if str(relative).is_empty() or not ResourceLoader.exists(path):
		return null
	var resource = load(path)
	return resource if resource is Texture2D else null


func _atlas_region(texture: Texture2D, cell: int, columns: int, cell_size: int) -> AtlasTexture:
	if texture == null or cell < 0 or columns < 1 or cell_size < 1:
		return null
	var x := (cell % columns) * cell_size
	var y := (cell / columns) * cell_size
	if x + cell_size > texture.get_width() or y + cell_size > texture.get_height():
		return null
	var region := AtlasTexture.new()
	region.atlas = texture
	region.region = Rect2(x, y, cell_size, cell_size)
	return region


func _family_filter_value() -> String:
	return ALL_FILTER if selected_family_filter == 0 else str(EXPECTED_FAMILIES[selected_family_filter - 1])


func _subtype_filter_value() -> String:
	if selected_subtype_filter == 0:
		return ALL_FILTER
	return "%s_%d" % [EXPECTED_FAMILIES[int((selected_subtype_filter - 1) / 4)], (selected_subtype_filter - 1) % 4]


func _role_filter_value() -> String:
	return ALL_FILTER if selected_role_filter == 0 else str(EXPECTED_ROLES[selected_role_filter - 1])


func _apply_filters(reset_selection := true) -> void:
	filtered_identities = []
	var family_filter := _family_filter_value()
	var subtype_filter := _subtype_filter_value()
	var role_filter := _role_filter_value()
	for value in identities:
		var identity: Dictionary = value
		if family_filter != ALL_FILTER and str(identity.get("family", "")) != family_filter:
			continue
		if subtype_filter != ALL_FILTER and str(identity.get("subtype", "")) != subtype_filter:
			continue
		if role_filter != ALL_FILTER and str(identity.get("role", "")) != role_filter:
			continue
		filtered_identities.append(identity)
	if reset_selection:
		selected_identity = 0
	selected_identity = clampi(selected_identity, 0, maxi(0, filtered_identities.size() - 1))
	_refresh_static(true)
	_refresh_motion(true)


func _selected_identity_record() -> Dictionary:
	if filtered_identities.is_empty():
		return {}
	return filtered_identities[selected_identity]


func _selected_motion_identity() -> Dictionary:
	var identity := _selected_identity_record()
	return motion_identities.get(str(identity.get("family", "")), {})


func _selected_motion_clip() -> Dictionary:
	var motion_identity := _selected_motion_identity()
	if motion_identity.is_empty():
		return {}
	var motion_name := str(EXPECTED_MOTIONS[selected_motion])
	var facing_name := str(EXPECTED_FACINGS[selected_facing])
	for value in motion_identity.get("clips", []):
		var clip: Dictionary = value
		if str(clip.get("motion", "")) == motion_name and str(clip.get("facing", "")) == facing_name:
			return clip
	return {}


func _motion_playback_frame_count(clip: Dictionary) -> int:
	var stored_count := maxi(1, int(clip.get("frame_count", 1)))
	# Looping clips store an exact duplicate endpoint as replay evidence. Playback
	# omits that terminal proof frame so the first pose is not held twice.
	return stored_count - 1 if bool(clip.get("loop", false)) and stored_count > 1 else stored_count


func _selected_map_entry() -> Dictionary:
	return map_entries.get(str(EXPECTED_THEMES[selected_theme]), {})


func _selected_map_layer_record() -> Dictionary:
	var entry := _selected_map_entry()
	var name := str(EXPECTED_MAP_LAYERS[selected_map_layer])
	for value in entry.get("layers", []):
		var layer: Dictionary = value
		if str(layer.get("name", "")) == name:
			return layer
	return {}


func _refresh_all() -> void:
	_refresh_static(true)
	_refresh_motion(true)
	_refresh_map(true)
	queue_redraw()


func _refresh_static(reload_atlas := false) -> void:
	var identity := _selected_identity_record()
	var layer_name := str(STATIC_LAYERS[selected_layer])
	filter_label.text = "FAMILY %s\nSUBTYPE %s\nROLE %s\nMATCHES %d / %d" % [_family_filter_value().to_upper(), _subtype_filter_value().to_upper(), _role_filter_value().to_upper(), filtered_identities.size(), identities.size()]
	layer_label.text = "LAYER\n" + layer_name.to_upper().replace("_", " ")
	scale_label.text = "VIEW %dx NATIVE" % selected_scale
	if identity.is_empty():
		static_rect.texture = null
		identity_label.text = "NO MATCH"
		static_hash_label.text = "Adjust family / subtype / role filters."
		return
	var atlas: Dictionary = static_atlases.get(layer_name, {})
	if reload_atlas or static_atlas_texture == null:
		static_atlas_texture = _load_texture(atlas.get("path", ""))
	static_rect.texture = _atlas_region(static_atlas_texture, int(identity.get("cell", -1)), int(atlas.get("columns", 0)), int(atlas.get("cell_size", 0)))
	var native_pixels := 48 * selected_scale
	static_rect.size = Vector2(native_pixels, native_pixels)
	static_rect.position = Vector2(16 + (256 - native_pixels) / 2.0, 39 + (256 - native_pixels) / 2.0)
	identity_label.text = "%s\n%s\n%s\nV%02d\n#%02d/%02d" % [str(identity.get("family", "")).to_upper(), str(identity.get("subtype", "")).to_upper(), str(identity.get("role", "")).to_upper(), int(identity.get("variant", 0)), selected_identity + 1, filtered_identities.size()]
	static_hash_label.text = "ID %s\nRAW %s\nCOMPILED %s\nLAYER %s\nATLAS %s" % [identity.get("sample_id", "missing"), str(identity.get("raw_fields_sha256", "missing")), str(identity.get("compiled_fields_sha256", "missing")), str(identity.get("source_layer_sha256", {}).get(layer_name, "missing")), str(atlas.get("sha256", "missing"))]


func _refresh_motion(reload_atlas := false) -> void:
	var identity := _selected_identity_record()
	if not _motion_available():
		motion_rect.texture = null
		var rejected: bool = motion_data.get("status", "") == "rejected"
		motion_status_label.text = "REJECTED\nFAIL-CLOSED" if rejected else "STAGED\nFAIL-CLOSED"
		motion_status_label.modulate = ERROR if rejected else WARN
		motion_label.text = str(EXPECTED_MOTIONS[selected_motion]).to_upper().replace("_", " ")
		facing_label.text = str(EXPECTED_FACINGS[selected_facing]).to_upper()
		motion_frame_label.text = "FRAME --"
		var reasons: Array = motion_data.get("reasons", [])
		motion_hash_label.text = "No procedural fallback is shown.\n\nExpected public neural-motion bank:\n%s\n\n%s" % [motion_data.get("expected", {}).get("bank_format", "missing"), "\n".join(reasons)]
		return
	var motion_identity := _selected_motion_identity()
	var clip := _selected_motion_clip()
	if identity.is_empty() or motion_identity.is_empty() or clip.is_empty():
		motion_rect.texture = null
		motion_status_label.text = "NO CLIP"
		return
	var layer_name := str(STATIC_LAYERS[selected_layer])
	var layer_record: Dictionary = motion_identity.get("layers", {}).get(layer_name, {})
	if reload_atlas or motion_atlas_texture == null:
		motion_atlas_texture = _load_texture(layer_record.get("path", ""))
	var playback_frame_count := _motion_playback_frame_count(clip)
	selected_motion_frame = posmod(selected_motion_frame, playback_frame_count)
	var cell := int(clip.get("start_cell", 0)) + selected_motion_frame
	var layout: Dictionary = motion_identity.get("layout", {})
	motion_rect.texture = _atlas_region(motion_atlas_texture, cell, int(layout.get("columns", 0)), int(layout.get("cell_size", 0)))
	var native_pixels := 48 * selected_scale
	motion_rect.size = Vector2(native_pixels, native_pixels)
	motion_rect.position = Vector2(20 + (256 - native_pixels) / 2.0, 42 + (256 - native_pixels) / 2.0)
	motion_status_label.text = "ONLINE\nNEURAL"
	motion_status_label.modulate = LIME
	motion_label.text = str(clip.get("motion", "")).to_upper().replace("_", " ")
	facing_label.text = str(clip.get("facing", "")).to_upper()
	motion_frame_label.text = "%02d / %02d%s" % [selected_motion_frame + 1, playback_frame_count, " LOOP" if bool(clip.get("loop", false)) else ""]
	var selected_static_id := str(identity.get("sample_id", "missing"))
	var representative_id := str(motion_identity.get("representative_static_sample_id", "missing"))
	motion_hash_label.text = "FAMILY REPRESENTATIVE %s  STATIC CELL %s\nCURRENT STATIC %s%s\nLAYER %s\nSOURCE CLIP %s\nDERIVED CLIP %s\nATLAS %s" % [representative_id, motion_identity.get("representative_static_cell", "?"), selected_static_id, " (EXACT REPRESENTATIVE)" if selected_static_id == representative_id else " (FAMILY PREVIEW USES REPRESENTATIVE)", layer_name, clip.get("source_clip_sha256", "missing"), clip.get("derived_clip_sha256", "missing"), layer_record.get("sha256", "missing")]


func _refresh_map(reload_atlas := false) -> void:
	var entry := _selected_map_entry()
	var layer := _selected_map_layer_record()
	if entry.is_empty() or layer.is_empty():
		map_rect.texture = null
		return
	if reload_atlas or map_atlas_texture == null:
		map_atlas_texture = _load_texture(entry.get("atlas", {}).get("path", ""))
	selected_map_frame = posmod(selected_map_frame, int(layer.get("frame_count", 1)))
	var cell := int(layer.get("start_cell", 0)) + selected_map_frame
	map_rect.texture = _atlas_region(map_atlas_texture, cell, int(entry.get("columns", 0)), int(entry.get("cell_size", 0)))
	theme_label.text = str(entry.get("theme", "")).to_upper()
	map_layer_label.text = str(layer.get("name", "")).to_upper().replace("_", " ")
	map_frame_label.text = "%02d/%02d" % [selected_map_frame + 1, layer.get("frame_count", 1)]
	var topology: Dictionary = entry.get("topology_contract", {})
	var stats: Dictionary = entry.get("statistics", {})
	map_meta_label.text = "MAP %s\nTOPOLOGY v%s  %s INVARIANTS\nPATH %s  BACKBONE SEGMENTS %s\nHAZARD %s  COLLISION %s  EMISSIVE %s" % [entry.get("map_id", "missing"), topology.get("schema_version", "?"), topology.get("invariant_count", "?"), topology.get("start_exit_path_length", "?"), topology.get("protected_backbone_segments", "?"), stats.get("animated_hazard_cells", "?"), stats.get("collision_cells", "?"), stats.get("emissive_pixels", "?")]
	map_hash_label.text = "SEMANTIC %s\nTOPOLOGY %s\nATLAS %s" % [entry.get("source_semantic_sha256", "missing"), entry.get("topology_manifest", {}).get("sha256", "missing"), entry.get("atlas", {}).get("sha256", "missing")]


func _step_identity(delta: int) -> void:
	if filtered_identities.is_empty(): return
	selected_identity = posmod(selected_identity + delta, filtered_identities.size())
	selected_motion_frame = 0
	_refresh_static()
	_refresh_motion(true)


func _step_layer(delta: int) -> void:
	selected_layer = posmod(selected_layer + delta, STATIC_LAYERS.size())
	_refresh_static(true)
	_refresh_motion(true)


func _step_family_filter(delta: int) -> void:
	selected_family_filter = posmod(selected_family_filter + delta, EXPECTED_FAMILIES.size() + 1)
	_apply_filters(true)


func _step_subtype_filter(delta: int) -> void:
	selected_subtype_filter = posmod(selected_subtype_filter + delta, 21)
	_apply_filters(true)


func _step_role_filter(delta: int) -> void:
	selected_role_filter = posmod(selected_role_filter + delta, EXPECTED_ROLES.size() + 1)
	_apply_filters(true)


func _clear_filters() -> void:
	selected_family_filter = 0
	selected_subtype_filter = 0
	selected_role_filter = 0
	_apply_filters(true)


func _toggle_scale() -> void:
	selected_scale = 1 if selected_scale == 4 else 4
	_refresh_static()
	_refresh_motion()


func _step_motion(delta: int) -> void:
	selected_motion = posmod(selected_motion + delta, EXPECTED_MOTIONS.size())
	selected_motion_frame = 0
	motion_accumulator = 0.0
	_refresh_motion()


func _step_facing(delta: int) -> void:
	selected_facing = posmod(selected_facing + delta, EXPECTED_FACINGS.size())
	selected_motion_frame = 0
	motion_accumulator = 0.0
	_refresh_motion()


func _step_motion_frame(delta: int) -> void:
	var clip := _selected_motion_clip()
	if clip.is_empty(): return
	selected_motion_frame = posmod(selected_motion_frame + delta, _motion_playback_frame_count(clip))
	motion_accumulator = 0.0
	_refresh_motion()


func _toggle_play() -> void:
	playing = not playing
	pause_button.text = "PAUSE" if playing else "PLAY"


func _step_theme(delta: int) -> void:
	selected_theme = posmod(selected_theme + delta, EXPECTED_THEMES.size())
	selected_map_frame = 0
	map_accumulator = 0.0
	_refresh_map(true)


func _step_map_layer(delta: int) -> void:
	selected_map_layer = posmod(selected_map_layer + delta, EXPECTED_MAP_LAYERS.size())
	selected_map_frame = 0
	map_accumulator = 0.0
	_refresh_map()


func _step_map_frame(delta: int) -> void:
	var layer := _selected_map_layer_record()
	if layer.is_empty(): return
	selected_map_frame = posmod(selected_map_frame + delta, int(layer.get("frame_count", 1)))
	map_accumulator = 0.0
	_refresh_map()


func _unhandled_key_input(event: InputEvent) -> void:
	if not event.is_pressed() or event.is_echo(): return
	match event.physical_keycode:
		KEY_Q: _step_identity(-1)
		KEY_E: _step_identity(1)
		KEY_1: _step_family_filter(1)
		KEY_2: _step_subtype_filter(1)
		KEY_3: _step_role_filter(1)
		KEY_0: _clear_filters()
		KEY_Z: _step_layer(-1)
		KEY_X: _step_layer(1)
		KEY_V: _toggle_scale()
		KEY_W: _step_motion(-1)
		KEY_S: _step_motion(1)
		KEY_A: _step_facing(-1)
		KEY_D: _step_facing(1)
		KEY_LEFT: _step_motion_frame(-1)
		KEY_RIGHT: _step_motion_frame(1)
		KEY_SPACE: _toggle_play()
		KEY_R: _step_theme(-1)
		KEY_F: _step_theme(1)
		KEY_T: _step_map_layer(-1)
		KEY_G: _step_map_layer(1)
		KEY_COMMA: _step_map_frame(-1)
		KEY_PERIOD: _step_map_frame(1)


func _process(delta: float) -> void:
	if not playing:
		return
	var clip := _selected_motion_clip()
	if _motion_available() and not clip.is_empty():
		motion_accumulator += delta
		var interval := 1.0 / maxf(1.0, float(clip.get("fps", 1)))
		if motion_accumulator >= interval:
			motion_accumulator = fmod(motion_accumulator, interval)
			var next := selected_motion_frame + 1
			if next >= _motion_playback_frame_count(clip):
				next = 0 if bool(clip.get("loop", false)) else selected_motion_frame
			if next != selected_motion_frame:
				selected_motion_frame = next
				_refresh_motion()
	var map_layer := _selected_map_layer_record()
	var fps := float(map_layer.get("fps", 0.0))
	if not map_layer.is_empty() and fps > 0.0:
		map_accumulator += delta
		if map_accumulator >= 1.0 / fps:
			map_accumulator = fmod(map_accumulator, 1.0 / fps)
			selected_map_frame = posmod(selected_map_frame + 1, int(map_layer.get("frame_count", 1)))
			_refresh_map()


func _draw() -> void:
	draw_rect(Rect2(0, 0, 1280, 720), PANEL_DEEP)
	for x in range(0, 1281, 32):
		var alpha := 0.07 if x % 128 == 0 else 0.023
		draw_line(Vector2(x, 74), Vector2(x, 674), Color(0.15, 0.6, 0.78, alpha), 1.0)
	for y in range(74, 675, 32):
		var alpha := 0.07 if y % 128 == 10 else 0.023
		draw_line(Vector2(0, y), Vector2(1280, y), Color(0.15, 0.6, 0.78, alpha), 1.0)
	draw_line(Vector2(22, 74), Vector2(1258, 74), RULE, 1.0)


func _smoke_option(prefix: String) -> String:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with(prefix):
			return argument.trim_prefix(prefix)
	return ""


func _check_hash(relative: Variant, expected: Variant, label: String, errors: Array[String]) -> void:
	var path := _asset_path(relative)
	if not FileAccess.file_exists(path):
		errors.append("missing " + label)
		return
	if FileAccess.get_sha256(path) != str(expected):
		errors.append("hash " + label)


func _write_smoke_report(path: String, report: Dictionary, errors: Array[String]) -> void:
	if path.is_empty(): return
	var handle := FileAccess.open(path, FileAccess.WRITE)
	if handle == null:
		errors.append("report write")
		return
	handle.store_string(JSON.stringify(report, "  ") + "\n")
	handle.close()


func _run_smoke_if_requested() -> void:
	if "--neural-workshop-smoke" not in OS.get_cmdline_user_args():
		return
	var errors: Array[String] = startup_errors.duplicate()
	var hash_checks := 0
	var static_regions := 0
	var map_regions := 0
	var map_frames := 0
	var motion_clips := 0
	var motion_frames := 0
	var motion_atlases := 0
	var motion_atlas_regions := 0
	if asset_index.get("format", "") != INDEX_FORMAT: errors.append("index format")
	if asset_index.get("engine", "") != "Godot 4.3": errors.append("engine")
	if asset_index.get("pixel_filter", "") != "nearest": errors.append("nearest index")
	var native_scales: Array = asset_index.get("native_scale_options", [])
	if native_scales.size() != 2 or int(native_scales[0]) != 1 or int(native_scales[1]) != 4: errors.append("native scales")
	if bool(asset_index.get("python_runtime_required", true)): errors.append("runtime Python")
	if identities.size() != 80: errors.append("identity count")
	if static_atlases.size() != 7: errors.append("static atlas count")
	if map_entries.size() != 6: errors.append("map count")
	if static_rect.texture_filter != CanvasItem.TEXTURE_FILTER_NEAREST or motion_rect.texture_filter != CanvasItem.TEXTURE_FILTER_NEAREST or map_rect.texture_filter != CanvasItem.TEXTURE_FILTER_NEAREST:
		errors.append("UI nearest filter")
	if int(ProjectSettings.get_setting("rendering/textures/canvas_textures/default_texture_filter", -1)) != 0:
		errors.append("project nearest filter")
	var preservation: Dictionary = asset_index.get("preservation", {})
	if preservation.get("main_scene", "") != "res://Arena.tscn" or not bool(preservation.get("baseline_preserved", false)):
		errors.append("Arena preservation contract")
	for value in asset_index.get("inventory", []):
		var record: Dictionary = value
		_check_hash(record.get("path", ""), record.get("sha256", ""), "inventory " + str(record.get("path", "")), errors)
		hash_checks += 1
	var expected_inventory_count := 69 if _motion_available() else 27
	if int(asset_index.get("asset_count", -1)) != expected_inventory_count or hash_checks != expected_inventory_count:
		errors.append("runtime inventory count")
	for layer_name in STATIC_LAYERS:
		var atlas: Dictionary = static_atlases.get(layer_name, {})
		var texture := _load_texture(atlas.get("path", ""))
		if texture == null:
			errors.append("load static atlas " + layer_name)
			continue
		var size: Array = atlas.get("size", [])
		if size.size() != 2 or texture.get_width() != int(size[0]) or texture.get_height() != int(size[1]):
			errors.append("static atlas size " + layer_name)
		for identity in identities:
			var region := _atlas_region(texture, int(identity.get("cell", -1)), int(atlas.get("columns", 0)), int(atlas.get("cell_size", 0)))
			if region == null or region.get_width() != 48 or region.get_height() != 48:
				errors.append("static region " + layer_name)
				break
			static_regions += 1
	for theme_position in range(EXPECTED_THEMES.size()):
		selected_theme = theme_position
		var entry := _selected_map_entry()
		var texture := _load_texture(entry.get("atlas", {}).get("path", ""))
		if texture == null:
			errors.append("load map " + str(EXPECTED_THEMES[theme_position]))
			continue
		var atlas_size: Array = entry.get("atlas_size", [])
		if atlas_size.size() != 2 or texture.get_width() != int(atlas_size[0]) or texture.get_height() != int(atlas_size[1]):
			errors.append("map size " + str(EXPECTED_THEMES[theme_position]))
		var topology: Dictionary = entry.get("topology_contract", {})
		if topology.get("schema_version", "") != "2.0.0" or not bool(topology.get("all_invariants_passed", false)):
			errors.append("map topology " + str(EXPECTED_THEMES[theme_position]))
		for layer_position in range(EXPECTED_MAP_LAYERS.size()):
			selected_map_layer = layer_position
			var map_layer := _selected_map_layer_record()
			if map_layer.is_empty():
				errors.append("map layer matrix")
				continue
			for frame in range(int(map_layer.get("frame_count", 0))):
				var cell := int(map_layer.get("start_cell", -1)) + frame
				var region := _atlas_region(texture, cell, int(entry.get("columns", 0)), int(entry.get("cell_size", 0)))
				if region == null:
					errors.append("map region")
					break
				map_frames += 1
			map_regions += 1
	if _motion_available():
		for family in EXPECTED_FAMILIES:
			var motion_identity: Dictionary = motion_identities.get(family, {})
			var expected_representative: Array = EXPECTED_MOTION_REPRESENTATIVES.get(family, [])
			if expected_representative.size() != 2 or str(motion_identity.get("sample_id", "")) != str(expected_representative[0]) or str(motion_identity.get("representative_static_sample_id", "")) != str(expected_representative[0]) or int(motion_identity.get("representative_static_cell", -1)) != int(expected_representative[1]):
				errors.append("motion representative %s" % family)
			elif int(expected_representative[1]) >= identities.size() or str(identities[int(expected_representative[1])].get("sample_id", "")) != str(expected_representative[0]):
				errors.append("motion static mapping %s" % family)
			if motion_identity.get("source_identity_manifest_semantics", "") != SOURCE_IDENTITY_MANIFEST_SEMANTICS or not (motion_identity.get("source_identity_manifest_audit_copy", {}) is Dictionary):
				errors.append("motion source manifest semantics %s" % family)
			var layout: Dictionary = motion_identity.get("layout", {})
			if int(layout.get("columns", -1)) != 16 or int(layout.get("rows", -1)) != 59 or int(layout.get("cell_size", -1)) != 48 or int(layout.get("frame_count", -1)) != 944:
				errors.append("motion layout %s" % family)
			var region_texture: Texture2D = null
			for layer_name in STATIC_LAYERS:
				var layer_record: Dictionary = motion_identity.get("layers", {}).get(layer_name, {})
				var texture := _load_texture(layer_record.get("path", ""))
				if texture == null:
					errors.append("motion atlas %s/%s" % [family, layer_name])
					continue
				if texture.get_width() != 768 or texture.get_height() != 2832:
					errors.append("motion atlas dimensions %s/%s" % [family, layer_name])
				if layer_name == "composite": region_texture = texture
				motion_atlases += 1
			for clip in motion_identity.get("clips", []):
				for frame in range(int(clip.get("frame_count", 0))):
					var cell := int(clip.get("start_cell", -1)) + frame
					var region := _atlas_region(region_texture, cell, int(layout.get("columns", 0)), int(layout.get("cell_size", 0)))
					if region == null or region.get_width() != 48 or region.get_height() != 48:
						errors.append("motion region %s/%s" % [family, cell])
					else:
						motion_atlas_regions += 1
					motion_frames += 1
				motion_clips += 1
		if motion_atlases != 35: errors.append("motion atlas coverage")
		if motion_clips != 520: errors.append("motion clip coverage")
		if motion_frames != 4720 or motion_atlas_regions != 4720: errors.append("motion frame region coverage")
	else:
		if motion_data.get("status", "") not in ["staged", "rejected"]: errors.append("motion staged status")
		if bool(motion_data.get("available", true)) or bool(motion_data.get("neural_output", true)): errors.append("motion fail-closed availability")
		if not motion_data.get("identities", []).is_empty() or int(motion_data.get("clip_count", -1)) != 0 or int(motion_data.get("frame_count", -1)) != 0: errors.append("motion staged payload")
	selected_family_filter = 0
	selected_subtype_filter = 0
	selected_role_filter = 0
	selected_identity = 0
	selected_layer = STATIC_LAYERS.find("composite")
	selected_theme = EXPECTED_THEMES.find("anomaly")
	selected_map_layer = EXPECTED_MAP_LAYERS.find("composite")
	selected_map_frame = 0
	_apply_filters(true)
	_refresh_all()
	var report := {
		"format": "nullvector-neural-workshop-godot-smoke-v1",
		"status": "passed" if errors.is_empty() else "failed",
		"passed": errors.is_empty(),
		"engine": Engine.get_version_info().get("string", "unknown"),
		"display_server": DisplayServer.get_name(),
		"scene": scene_file_path,
		"index": INDEX_PATH,
		"bundle_id": asset_index.get("bundle_id", "missing"),
		"nearest_filtering": true,
		"python_runtime_required": false,
		"motion_status": motion_data.get("status", "missing"),
		"motion_fail_closed": not _motion_available(),
		"coverage": {
			"identities": identities.size(),
			"static_layers": static_atlases.size(),
			"static_regions": static_regions,
			"map_themes": map_entries.size(),
			"map_layers": map_regions,
			"map_frames": map_frames,
			"motion_atlases": motion_atlases,
			"motion_clips": motion_clips,
			"motion_frames": motion_frames,
			"motion_atlas_regions": motion_atlas_regions,
			"runtime_hashes": hash_checks,
		},
		"errors": errors,
	}
	_write_smoke_report(_smoke_option("--neural-workshop-report="), report, errors)
	if errors.is_empty():
		print("NEURAL_WORKSHOP_SMOKE_OK identities=%d static_layers=%d static_regions=%d maps=%d map_regions=%d map_frames=%d motion=%s motion_atlases=%d motion_clips=%d motion_frames=%d motion_atlas_regions=%d hashes=%d" % [identities.size(), static_atlases.size(), static_regions, map_entries.size(), map_regions, map_frames, motion_data.get("status", "missing"), motion_atlases, motion_clips, motion_frames, motion_atlas_regions, hash_checks])
		get_tree().quit(0)
	else:
		push_error("NEURAL_WORKSHOP_SMOKE_FAILED " + ", ".join(errors))
		get_tree().quit(1)
