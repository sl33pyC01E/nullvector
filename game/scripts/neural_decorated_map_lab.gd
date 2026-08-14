extends Node2D

const FORMAT := "nullvector-neural-decorated-map-native-catalog/1.0.0"
const THEMES := ["arena", "rooms", "caves", "archipelago", "garden", "anomaly"]
const LAYERS := ["composite", "base_color", "emissive", "objects", "variant", "emission_level", "topology", "hazard"]
const AUTHORITY := {
	"variant": "deterministic_semantic_teacher",
	"decal": "accepted_neural_protected_selector",
	"prop": "accepted_neural_protected_selector",
	"emission": "conditional_semantic_projection",
}
const BACKGROUND := Color("#03080d")
const PANEL := Color("#07121b")
const RULE := Color("#183243")
const CYAN := Color("#38e8ff")
const MAGENTA := Color("#ff36c8")
const LIME := Color("#9bff4f")
const MUTED := Color("#79909d")
const WARN := Color("#ff526d")

@export_file("*.json") var catalog_path := "res://generated/neural_decorated_maps/v1_1/catalog.json"
@export_dir var asset_root := "res://generated/neural_decorated_maps/v1_1/"

var catalog: Dictionary = {}
var maps: Array = []
var atlas_texture: Texture2D
var startup_errors: Array[String] = []
var theme_index := 0
var layer_index := 0
var frame_index := 0
var frame_accumulator := 0.0
var playing := true
var map_rect: TextureRect
var title_label: Label
var status_label: Label
var theme_label: Label
var layer_label: Label
var frame_label: Label
var authority_label: Label
var provenance_label: Label
var pause_button: Button


func _ready() -> void:
	catalog = _load_json(catalog_path)
	_validate_catalog()
	_build_ui()
	_refresh_map()
	if not startup_errors.is_empty():
		status_label.text = "FAIL-CLOSED // " + ", ".join(startup_errors)
		status_label.modulate = WARN
	if "--neural-decorated-map-smoke" in OS.get_cmdline_user_args() or not _smoke_option("--neural-decorated-map-report=").is_empty():
		call_deferred("_run_smoke")


func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path):
		startup_errors.append("missing catalog")
		return {}
	var parsed = JSON.parse_string(FileAccess.get_file_as_string(path))
	if not parsed is Dictionary:
		startup_errors.append("catalog json")
		return {}
	return parsed


func _validate_catalog() -> void:
	if catalog.get("format", "") != FORMAT: startup_errors.append("catalog format")
	if catalog.get("status", "") != "ready": startup_errors.append("catalog status")
	if catalog.get("engine", "") != "Godot 4.3": startup_errors.append("engine contract")
	if catalog.get("themes", []) != THEMES: startup_errors.append("theme vocabulary")
	if catalog.get("layers", []) != LAYERS: startup_errors.append("layer vocabulary")
	if int(catalog.get("theme_count", -1)) != 6 or int(catalog.get("layer_count", -1)) != 8: startup_errors.append("catalog census")
	if int(catalog.get("atlas_frame_count", -1)) != 90 or int(catalog.get("hazard_frames_per_theme", -1)) != 8: startup_errors.append("frame census")
	if bool(catalog.get("python_runtime_required", true)) or bool(catalog.get("checkpoint_shipped", true)): startup_errors.append("runtime boundary")
	if catalog.get("runtime_asset_extensions", []) != [".json", ".png"]: startup_errors.append("runtime extensions")
	if not bool(catalog.get("visual_inspection_passed", false)): startup_errors.append("visual inspection")
	maps = catalog.get("maps", [])
	if maps.size() != 6: startup_errors.append("map census")
	for position in range(maps.size()):
		var entry: Dictionary = maps[position]
		if str(entry.get("theme", "")) != THEMES[position]: startup_errors.append("map order")
		if entry.get("layers", []).size() != 8: startup_errors.append("map layer census")
		var selection: Dictionary = entry.get("selection", {})
		if selection.get("field_authority", {}) != AUTHORITY: startup_errors.append("field authority")
		if bool(selection.get("unsupported_neural_heads_cross_runtime_boundary", true)): startup_errors.append("unsupported neural heads")
		if not bool(selection.get("validation", {}).get("passed", false)): startup_errors.append("selection legality")
	var atlas: Dictionary = catalog.get("atlas", {})
	var path := asset_root + str(atlas.get("path", ""))
	if not FileAccess.file_exists(path):
		startup_errors.append("atlas missing")
		return
	if FileAccess.get_file_as_bytes(path).size() != int(atlas.get("bytes", -1)) or FileAccess.get_sha256(path) != str(atlas.get("sha256", "")):
		startup_errors.append("atlas identity")
	var resource = load(path)
	if not resource is Texture2D:
		startup_errors.append("atlas texture")
		return
	atlas_texture = resource
	var size: Array = atlas.get("size", [])
	if size.size() != 2 or atlas_texture.get_width() != int(size[0]) or atlas_texture.get_height() != int(size[1]): startup_errors.append("atlas size")
	if int(atlas.get("columns", -1)) != 4 or int(atlas.get("rows", -1)) != 23 or int(atlas.get("cell_size", -1)) != 384: startup_errors.append("atlas grid")
	if int(ProjectSettings.get_setting("rendering/textures/canvas_textures/default_texture_filter", -1)) != 0: startup_errors.append("project nearest filter")


func _font_label(parent: Node, position: Vector2, size: Vector2, text: String, color: Color, font_size: int) -> Label:
	var label := Label.new()
	label.position = position
	label.size = size
	label.text = text
	label.modulate = color
	label.add_theme_font_size_override("font_size", font_size)
	parent.add_child(label)
	return label


func _button(parent: Node, position: Vector2, text: String, callback: Callable) -> Button:
	var button := Button.new()
	button.position = position
	button.size = Vector2(112, 34)
	button.text = text
	button.pressed.connect(callback)
	parent.add_child(button)
	return button


func _build_ui() -> void:
	var canvas := CanvasLayer.new()
	add_child(canvas)
	var header := ColorRect.new()
	header.color = PANEL
	header.position = Vector2(24, 20)
	header.size = Vector2(1232, 62)
	canvas.add_child(header)
	title_label = _font_label(header, Vector2(18, 8), Vector2(780, 26), "NULLVECTOR // NEURAL DECORATED MAP LAB", CYAN, 18)
	status_label = _font_label(header, Vector2(18, 34), Vector2(950, 20), "READY // ACCEPTED OBJECT HEADS + SEMANTIC TERRAIN AUTHORITY", LIME, 10)
	_font_label(header, Vector2(1015, 18), Vector2(195, 24), "6 THEMES // 90 FRAMES", MAGENTA, 11)

	var viewport_panel := ColorRect.new()
	viewport_panel.color = PANEL
	viewport_panel.position = Vector2(24, 98)
	viewport_panel.size = Vector2(704, 598)
	canvas.add_child(viewport_panel)
	map_rect = TextureRect.new()
	map_rect.position = Vector2(34, 24)
	map_rect.size = Vector2(540, 540)
	map_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	map_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	map_rect.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	map_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
	viewport_panel.add_child(map_rect)
	theme_label = _font_label(viewport_panel, Vector2(590, 40), Vector2(102, 54), "THEME", MAGENTA, 14)
	layer_label = _font_label(viewport_panel, Vector2(590, 110), Vector2(102, 70), "LAYER", CYAN, 12)
	frame_label = _font_label(viewport_panel, Vector2(590, 200), Vector2(102, 30), "FRAME", MUTED, 10)
	_button(viewport_panel, Vector2(582, 258), "THEME -", func(): _step_theme(-1))
	_button(viewport_panel, Vector2(582, 298), "THEME +", func(): _step_theme(1))
	_button(viewport_panel, Vector2(582, 354), "LAYER -", func(): _step_layer(-1))
	_button(viewport_panel, Vector2(582, 394), "LAYER +", func(): _step_layer(1))
	pause_button = _button(viewport_panel, Vector2(582, 450), "PAUSE", _toggle_play)

	var audit_panel := ColorRect.new()
	audit_panel.color = PANEL
	audit_panel.position = Vector2(746, 98)
	audit_panel.size = Vector2(510, 598)
	canvas.add_child(audit_panel)
	_font_label(audit_panel, Vector2(22, 18), Vector2(460, 24), "FIELD AUTHORITY", MAGENTA, 13)
	authority_label = _font_label(audit_panel, Vector2(22, 54), Vector2(460, 210), "", MUTED, 11)
	authority_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_font_label(audit_panel, Vector2(22, 278), Vector2(460, 24), "REPLAY + PROVENANCE", CYAN, 13)
	provenance_label = _font_label(audit_panel, Vector2(22, 314), Vector2(460, 245), "", MUTED, 9)
	provenance_label.autowrap_mode = TextServer.AUTOWRAP_ARBITRARY


func _selected_map() -> Dictionary:
	return maps[theme_index] if theme_index >= 0 and theme_index < maps.size() else {}


func _selected_layer() -> Dictionary:
	var entry := _selected_map()
	if entry.is_empty(): return {}
	var layers: Array = entry.get("layers", [])
	return layers[layer_index] if layer_index >= 0 and layer_index < layers.size() else {}


func _refresh_map() -> void:
	var entry := _selected_map()
	var layer := _selected_layer()
	if entry.is_empty() or layer.is_empty() or atlas_texture == null:
		map_rect.texture = null
		return
	frame_index = posmod(frame_index, maxi(1, int(layer.get("frame_count", 1))))
	var cell := int(layer.get("start_cell", 0)) + frame_index
	var columns := int(catalog.get("atlas", {}).get("columns", 4))
	var cell_size := int(catalog.get("atlas", {}).get("cell_size", 384))
	var region := AtlasTexture.new()
	region.atlas = atlas_texture
	region.region = Rect2((cell % columns) * cell_size, (cell / columns) * cell_size, cell_size, cell_size)
	map_rect.texture = region
	theme_label.text = str(entry.get("theme", "?")).to_upper()
	layer_label.text = str(layer.get("name", "?")).to_upper().replace("_", " ")
	frame_label.text = "%02d / %02d\n%s FPS" % [frame_index + 1, int(layer.get("frame_count", 1)), layer.get("fps", 0)]
	var selection: Dictionary = entry.get("selection", {})
	var counts: Dictionary = selection.get("validation", {}).get("counts", {})
	authority_label.text = "VARIANT  SEMANTIC HASH\nDECAL    ACCEPTED NEURAL\nPROP     ACCEPTED NEURAL\nEMISSION CONDITIONAL SEMANTIC\n\nOBJECTS  D%s / P%s\nEMISSIVE %s CELLS\nINSTANCES %s\n\nUNSUPPORTED NEURAL HEADS: BLOCKED" % [counts.get("decal", "?"), counts.get("prop", "?"), counts.get("emission", "?"), entry.get("instance_count", "?")]
	provenance_label.text = "MAP %s\nSOURCE %s\nTOPOLOGY %s\nFIELDS %s\nATLAS %s\nBUNDLE %s" % [entry.get("map_id", "?"), entry.get("source_semantic_sha256", "?"), entry.get("topology_masks_sha256", "?"), selection.get("selection_fields_sha256", "?"), catalog.get("atlas", {}).get("sha256", "?"), catalog.get("bundle_id", "?")]


func _step_theme(delta: int) -> void:
	theme_index = posmod(theme_index + delta, THEMES.size())
	frame_index = 0
	frame_accumulator = 0.0
	_refresh_map()


func _step_layer(delta: int) -> void:
	layer_index = posmod(layer_index + delta, LAYERS.size())
	frame_index = 0
	frame_accumulator = 0.0
	_refresh_map()


func _toggle_play() -> void:
	playing = not playing
	pause_button.text = "PAUSE" if playing else "PLAY"


func _unhandled_input(event: InputEvent) -> void:
	if not event is InputEventKey or not event.pressed or event.echo: return
	match (event as InputEventKey).keycode:
		KEY_LEFT: _step_theme(-1)
		KEY_RIGHT: _step_theme(1)
		KEY_UP: _step_layer(-1)
		KEY_DOWN: _step_layer(1)
		KEY_SPACE: _toggle_play()


func _process(delta: float) -> void:
	if not playing: return
	var layer := _selected_layer()
	var fps := float(layer.get("fps", 0.0))
	if fps <= 0.0: return
	frame_accumulator += delta
	var interval := 1.0 / fps
	if frame_accumulator >= interval:
		frame_accumulator = fmod(frame_accumulator, interval)
		frame_index = posmod(frame_index + 1, int(layer.get("frame_count", 1)))
		_refresh_map()


func _draw() -> void:
	draw_rect(Rect2(0, 0, 1280, 720), BACKGROUND)
	for x in range(0, 1281, 32): draw_line(Vector2(x, 84), Vector2(x, 720), Color(0.12, 0.45, 0.62, 0.035), 1.0)
	for y in range(84, 721, 32): draw_line(Vector2(0, y), Vector2(1280, y), Color(0.12, 0.45, 0.62, 0.035), 1.0)
	draw_line(Vector2(24, 86), Vector2(1256, 86), RULE, 1.0)


func _smoke_option(prefix: String) -> String:
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with(prefix): return argument.trim_prefix(prefix)
	return ""


func _run_smoke() -> void:
	var errors: Array[String] = startup_errors.duplicate()
	var regions := 0
	var frames := 0
	var authority_exact := true
	var atlas: Dictionary = catalog.get("atlas", {})
	var columns := int(atlas.get("columns", 0))
	var rows := int(atlas.get("rows", 0))
	var cell_size := int(atlas.get("cell_size", 0))
	if atlas_texture == null:
		errors.append("atlas unavailable")
	for theme_position in range(maps.size()):
		var entry: Dictionary = maps[theme_position]
		if entry.get("selection", {}).get("field_authority", {}) != AUTHORITY:
			authority_exact = false
			errors.append("authority %s" % entry.get("theme", "?"))
		for layer in entry.get("layers", []):
			var start := int(layer.get("start_cell", -1))
			var count := int(layer.get("frame_count", 0))
			for local_frame in range(count):
				var cell := start + local_frame
				var x := (cell % columns) * cell_size
				var y := (cell / columns) * cell_size
				if atlas_texture == null or cell < 0 or cell >= columns * rows or x + cell_size > atlas_texture.get_width() or y + cell_size > atlas_texture.get_height():
					errors.append("region bounds")
					continue
				var region := AtlasTexture.new()
				region.atlas = atlas_texture
				region.region = Rect2(x, y, cell_size, cell_size)
				if region.get_width() != cell_size or region.get_height() != cell_size: errors.append("region texture")
				frames += 1
			regions += 1
	if map_rect.texture_filter != CanvasItem.TEXTURE_FILTER_NEAREST: errors.append("ui nearest filter")
	if regions != 48 or frames != 90: errors.append("coverage totals")
	var report := {
		"format": "nullvector-neural-decorated-map-godot-smoke/1.0.0",
		"passed": errors.is_empty(),
		"engine": Engine.get_version_info().get("string", "unknown"),
		"scene": scene_file_path,
		"theme_count": maps.size(),
		"layer_count": LAYERS.size(),
		"regions_checked": regions,
		"frames_checked": frames,
		"atlas_size": [atlas_texture.get_width(), atlas_texture.get_height()] if atlas_texture != null else [],
		"field_authority_exact": authority_exact,
		"nearest_filtering": true,
		"python_runtime_required": false,
		"bundle_id": catalog.get("bundle_id", ""),
		"errors": errors,
	}
	var report_path := _smoke_option("--neural-decorated-map-report=")
	if not report_path.is_empty():
		var handle := FileAccess.open(report_path, FileAccess.WRITE)
		if handle == null: errors.append("report write")
		else: handle.store_string(JSON.stringify(report, "  ") + "\n"); handle.close()
	if errors.is_empty():
		print("NEURAL_DECORATED_MAP_SMOKE_OK themes=6 layers=8 regions=48 frames=90")
		get_tree().quit(0)
	else:
		push_error("NEURAL_DECORATED_MAP_SMOKE_FAILED " + ", ".join(errors))
		get_tree().quit(1)
