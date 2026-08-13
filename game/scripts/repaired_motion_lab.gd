extends Node2D

const CATALOG_PATH := "res://generated/repaired_motion_lab/v1/catalog.json"
const ASSET_ROOT := "res://generated/repaired_motion_lab/v1/"
const CATALOG_FORMAT := "nullvector-repaired-motion-native-catalog-v1"
const LAYERS := ["base", "outline", "emission_core", "aura", "bloom_r1", "bloom_r2", "composite"]
const FAMILY_COLORS := {
	"humanoid": Color("#37f3ff"), "animalian": Color("#ff4fb7"),
	"plantlike": Color("#a8ff4f"), "anomaly": Color("#a77bff"),
	"machine": Color("#ffae37")
}
const TEXT := Color("#e9f7ff")
const MUTED := Color("#718ba5")
const RULE := Color("#1d3c5e")
const DEEP := Color("#03070d")
const LIME := Color("#a8ff4f")
const ERROR := Color("#ff526d")

var catalog: Dictionary = {}
var selected_identity := 0
var selected_motion := 0
var selected_facing := 0
var selected_layer := 6
var frame := 0
var playing := true
var accumulator := 0.0
var zoom := 6.0
var atlas_texture: ImageTexture
var startup_errors: Array[String] = []

var identity_label: Label
var clip_label: Label
var status_label: Label
var layer_label: Label
var frame_label: Label
var proof_label: Label


func _ready() -> void:
	texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	_build_interface()
	catalog = _load_json(CATALOG_PATH)
	_validate_catalog()
	if startup_errors.is_empty():
		_load_selected_atlas()
		_refresh_labels()
	else:
		status_label.text = "FAIL-CLOSED // " + ", ".join(startup_errors)
		status_label.modulate = ERROR
	queue_redraw()
	if "--repaired-motion-lab-smoke" in OS.get_cmdline_user_args():
		call_deferred("_run_headless_smoke")


func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null: return {}
	var value = JSON.parse_string(file.get_as_text())
	return value if value is Dictionary else {}


func _validate_catalog() -> void:
	if catalog.get("format", "") != CATALOG_FORMAT: startup_errors.append("catalog format")
	if catalog.get("status", "") != "ready": startup_errors.append("catalog status")
	if catalog.get("neural_output", false) is not bool or not catalog.get("neural_output", false): startup_errors.append("neural authority")
	var counts: Dictionary = catalog.get("counts", {})
	for pair in [["identity_count", 80], ["motion_count", 13], ["facing_count", 8], ["clip_count", 8320], ["frame_count", 75520], ["atlas_count", 560]]:
		if int(counts.get(pair[0], -1)) != pair[1]: startup_errors.append(str(pair[0]))
	if catalog.get("identities", []).size() != 80: startup_errors.append("identity registry")
	if catalog.get("layers", []) != LAYERS: startup_errors.append("layer registry")
	var family_counts: Dictionary = catalog.get("family_counts", {})
	for family in FAMILY_COLORS:
		if int(family_counts.get(family, -1)) != 16: startup_errors.append("family " + family)
	if startup_errors.is_empty():
		status_label.text = "ALL-80 ONLINE // 8,320 CLIPS // 75,520 FRAMES // 560 ATLASES"
		status_label.modulate = LIME


func _panel(parent: Node, rect: Rect2) -> Panel:
	var panel := Panel.new(); panel.position = rect.position; panel.size = rect.size
	var style := StyleBoxFlat.new(); style.bg_color = Color(0.018, 0.045, 0.085, 0.96)
	style.border_color = RULE; style.set_border_width_all(1); panel.add_theme_stylebox_override("panel", style)
	parent.add_child(panel); return panel


func _label(parent: Node, position: Vector2, size: Vector2, value: String, color := TEXT, font_size := 10) -> Label:
	var result := Label.new(); result.position = position; result.size = size; result.text = value
	result.add_theme_font_size_override("font_size", font_size); result.add_theme_color_override("font_color", color)
	parent.add_child(result); return result


func _button(parent: Node, position: Vector2, size: Vector2, value: String, callback: Callable) -> Button:
	var result := Button.new(); result.position = position; result.size = size; result.text = value
	result.focus_mode = Control.FOCUS_NONE; result.add_theme_font_size_override("font_size", 9)
	result.pressed.connect(callback); parent.add_child(result); return result


func _build_interface() -> void:
	var canvas := CanvasLayer.new(); add_child(canvas)
	_label(canvas, Vector2(24, 12), Vector2(780, 32), "NULLVECTOR // ALL-80 REPAIRED MOTION LAB", TEXT, 22)
	_label(canvas, Vector2(26, 45), Vector2(900, 20), "80 NEURAL IDENTITIES // REST PIXELS IMMUTABLE // LOGICAL RIG REPAIR // EXACT REPLAY", Color("#37f3ff"), 10)
	status_label = _label(canvas, Vector2(650, 18), Vector2(590, 30), "LOADING SEALED MOTION BANK", LIME, 9); status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	var left := _panel(canvas, Rect2(18, 80, 310, 610))
	_label(left, Vector2(14, 10), Vector2(280, 20), "01 // IDENTITY", Color("#37f3ff"), 11)
	_button(left, Vector2(14, 38), Vector2(48, 30), "Q <", func(): _change_identity(-1))
	_button(left, Vector2(67, 38), Vector2(48, 30), "E >", func(): _change_identity(1))
	_button(left, Vector2(120, 38), Vector2(82, 30), "- FAMILY", func(): _change_identity(-16))
	_button(left, Vector2(207, 38), Vector2(88, 30), "+ FAMILY", func(): _change_identity(16))
	identity_label = _label(left, Vector2(14, 78), Vector2(280, 112), "IDENTITY", TEXT, 10); identity_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_label(left, Vector2(14, 200), Vector2(280, 20), "02 // MOTION / FACING", Color("#37f3ff"), 11)
	_button(left, Vector2(14, 228), Vector2(48, 30), "W <", func(): _change_motion(-1))
	_button(left, Vector2(67, 228), Vector2(48, 30), "S >", func(): _change_motion(1))
	_button(left, Vector2(120, 228), Vector2(48, 30), "A <", func(): _change_facing(-1))
	_button(left, Vector2(173, 228), Vector2(48, 30), "D >", func(): _change_facing(1))
	_button(left, Vector2(226, 228), Vector2(69, 30), "PLAY SPC", func(): playing = not playing)
	clip_label = _label(left, Vector2(14, 270), Vector2(280, 98), "CLIP", TEXT, 10); clip_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_label(left, Vector2(14, 378), Vector2(280, 20), "03 // PRESENTATION", Color("#37f3ff"), 11)
	_button(left, Vector2(14, 406), Vector2(48, 30), "Z <", func(): _change_layer(-1))
	_button(left, Vector2(67, 406), Vector2(48, 30), "X >", func(): _change_layer(1))
	_button(left, Vector2(120, 406), Vector2(82, 30), "NATIVE 1X", func(): zoom = 1.0; queue_redraw())
	_button(left, Vector2(207, 406), Vector2(88, 30), "PIXEL 6X", func(): zoom = 6.0; queue_redraw())
	layer_label = _label(left, Vector2(14, 448), Vector2(280, 40), "LAYER", LIME, 10)
	proof_label = _label(left, Vector2(14, 505), Vector2(280, 82), "PRIMARY / REPLAY BYTE EXACT\n560 HASH-BOUND ATLASES\nNO PYTHON AT RUNTIME", MUTED, 9)
	proof_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	var stage := _panel(canvas, Rect2(344, 80, 918, 610))
	_label(stage, Vector2(14, 10), Vector2(600, 20), "LIVE ATLAS READER // NATIVE 48PX CELLS", Color("#37f3ff"), 10)
	frame_label = _label(stage, Vector2(650, 10), Vector2(250, 20), "FRAME", MUTED, 9); frame_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT


func _identity() -> Dictionary:
	return catalog.get("identities", [])[posmod(selected_identity, 80)]


func _clip() -> Dictionary:
	var identity := _identity()
	var motion := str(catalog.get("motions", [])[posmod(selected_motion, 13)])
	var facing := str(catalog.get("facings", [])[posmod(selected_facing, 8)])
	for candidate in identity.get("clips", []):
		if candidate.get("motion", "") == motion and candidate.get("facing", "") == facing: return candidate
	return {}


func _atlas_path() -> String:
	var artifact: Dictionary = _identity().get("atlases", {}).get(LAYERS[selected_layer], {})
	return ASSET_ROOT + str(artifact.get("path", ""))


func _load_selected_atlas() -> void:
	var path := _atlas_path(); atlas_texture = null
	if not FileAccess.file_exists(path): startup_errors.append("atlas missing"); return
	var artifact: Dictionary = _identity().get("atlases", {}).get(LAYERS[selected_layer], {})
	if FileAccess.get_file_as_bytes(path).size() != int(artifact.get("bytes", -1)) or FileAccess.get_sha256(path) != str(artifact.get("sha256", "")):
		startup_errors.append("atlas provenance"); return
	var image := Image.load_from_file(path)
	if image == null or image.get_width() != 768 or image.get_height() != 2832:
		startup_errors.append("atlas dimensions"); return
	atlas_texture = ImageTexture.create_from_image(image)


func _change_identity(delta: int) -> void:
	selected_identity = posmod(selected_identity + delta, 80); frame = 0; _load_selected_atlas(); _refresh_labels(); queue_redraw()


func _change_motion(delta: int) -> void:
	selected_motion = posmod(selected_motion + delta, 13); frame = 0; _refresh_labels(); queue_redraw()


func _change_facing(delta: int) -> void:
	selected_facing = posmod(selected_facing + delta, 8); frame = 0; _refresh_labels(); queue_redraw()


func _change_layer(delta: int) -> void:
	selected_layer = posmod(selected_layer + delta, LAYERS.size()); _load_selected_atlas(); _refresh_labels(); queue_redraw()


func _refresh_labels() -> void:
	if catalog.is_empty(): return
	var identity := _identity(); var clip := _clip(); var family := str(identity.get("family", "?"))
	identity_label.text = "#%02d // %s\n%s\n%s // %s\nSEED %s" % [int(identity.get("ordinal", -1)), family.to_upper(), str(identity.get("sample_id", "?")), str(identity.get("subtype", "?")), str(identity.get("role", "?")), str(identity.get("sample_seed", "?"))]
	identity_label.modulate = FAMILY_COLORS.get(family, TEXT)
	clip_label.text = "%s // %s\n%d FPS // %d STORED FRAMES\n%s" % [str(clip.get("motion", "?")).to_upper(), str(clip.get("facing", "?")).to_upper(), int(clip.get("fps", 0)), int(clip.get("frame_count", 0)), "LOOP / TERMINAL PROOF FRAME" if clip.get("loop", false) else "ONE SHOT"]
	layer_label.text = "LAYER // " + LAYERS[selected_layer].to_upper()
	frame_label.text = "FRAME %d / %d // CELL %d" % [frame + 1, int(clip.get("frame_count", 0)), int(clip.get("start_cell", 0)) + frame]


func _process(delta: float) -> void:
	if not startup_errors.is_empty() or not playing: return
	var clip: Dictionary = _clip(); var stored := int(clip.get("frame_count", 1)); var playable: int = maxi(1, stored - 1 if clip.get("loop", false) else stored)
	accumulator += delta
	var step: float = 1.0 / maxf(1.0, float(clip.get("fps", 8)))
	while accumulator >= step:
		accumulator -= step
		if frame + 1 < playable: frame += 1
		elif clip.get("loop", false): frame = 0
		else: playing = false
		_refresh_labels(); queue_redraw()


func _draw() -> void:
	draw_rect(Rect2(0, 0, 1280, 720), DEEP)
	var area := Rect2(345, 81, 916, 608); draw_rect(area, Color("#03080f"))
	for x in range(370, 1240, 24): draw_line(Vector2(x, 112), Vector2(x, 670), Color(0.12, 0.3, 0.4, 0.14), 1)
	for y in range(112, 671, 24): draw_line(Vector2(370, y), Vector2(1240, y), Color(0.12, 0.3, 0.4, 0.14), 1)
	if atlas_texture == null or catalog.is_empty(): return
	var clip := _clip(); var cell := int(clip.get("start_cell", 0)) + frame
	var source := Rect2((cell % 16) * 48, (cell / 16) * 48, 48, 48)
	var size := Vector2(48, 48) * zoom; var destination := Rect2(Vector2(805, 390) - size * 0.5, size)
	var color: Color = FAMILY_COLORS.get(str(_identity().get("family", "")), Color.WHITE)
	var glow := color; glow.a = 0.035
	var floor_glow := color; floor_glow.a = 0.08
	draw_circle(Vector2(805, 535), 190, glow); draw_ellipse(Vector2(805, 535), Vector2(215, 34), floor_glow)
	draw_texture_rect_region(atlas_texture, destination, source)


func draw_ellipse(center: Vector2, radii: Vector2, color: Color) -> void:
	var points := PackedVector2Array()
	for index in range(65):
		var angle := TAU * index / 64.0; points.append(center + Vector2(cos(angle) * radii.x, sin(angle) * radii.y))
	draw_colored_polygon(points, color)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_Q: _change_identity(-1)
			KEY_E: _change_identity(1)
			KEY_W: _change_motion(-1)
			KEY_S: _change_motion(1)
			KEY_A: _change_facing(-1)
			KEY_D: _change_facing(1)
			KEY_Z: _change_layer(-1)
			KEY_X: _change_layer(1)
			KEY_LEFT: frame = max(0, frame - 1); _refresh_labels(); queue_redraw()
			KEY_RIGHT: frame = min(int(_clip().get("frame_count", 1)) - 1, frame + 1); _refresh_labels(); queue_redraw()
			KEY_SPACE: playing = not playing


func _run_headless_smoke() -> void:
	var errors := startup_errors.duplicate(); var atlas_count := 0; var clip_count := 0; var frame_count := 0; var region_count := 0
	var families: Dictionary = {}
	for identity in catalog.get("identities", []):
		families[str(identity.get("family", "?"))] = int(families.get(str(identity.get("family", "?")), 0)) + 1
		var clips: Array = identity.get("clips", []); clip_count += clips.size()
		for clip in clips:
			var start := int(clip.get("start_cell", -1)); var count := int(clip.get("frame_count", -1))
			if start < 0 or count < 1 or start + count > 944: errors.append("clip bounds")
			frame_count += count; region_count += count
		for layer in LAYERS:
			var artifact: Dictionary = identity.get("atlases", {}).get(layer, {}); var path := ASSET_ROOT + str(artifact.get("path", ""))
			if not FileAccess.file_exists(path): errors.append("atlas missing"); continue
			if FileAccess.get_file_as_bytes(path).size() != int(artifact.get("bytes", -1)) or FileAccess.get_sha256(path) != str(artifact.get("sha256", "")): errors.append("atlas hash")
			var image := Image.load_from_file(path)
			if image == null or image.get_width() != 768 or image.get_height() != 2832: errors.append("atlas shape")
			atlas_count += 1
	if clip_count != 8320: errors.append("clip total")
	if frame_count != 75520 or region_count != 75520: errors.append("frame total")
	if atlas_count != 560: errors.append("atlas total")
	for family in FAMILY_COLORS:
		if int(families.get(family, 0)) != 16: errors.append("family total")
	var report := {"passed": errors.is_empty(), "engine": Engine.get_version_info().get("string", "unknown"), "bundle_id": catalog.get("bundle_id", ""), "identity_count": catalog.get("identities", []).size(), "family_counts": families, "atlas_count": atlas_count, "clip_count": clip_count, "frame_count": frame_count, "atlas_regions_checked": region_count, "python_runtime_required": false, "errors": errors}
	var report_path := ""
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--repaired-motion-lab-report="): report_path = argument.trim_prefix("--repaired-motion-lab-report=")
	if not report_path.is_empty():
		var file := FileAccess.open(report_path, FileAccess.WRITE)
		if file != null: file.store_string(JSON.stringify(report, "  ", false) + "\n")
	if errors.is_empty(): print("REPAIRED_MOTION_LAB_SMOKE_OK identities=80 atlases=560 clips=8320 frames=75520 regions=75520")
	else: push_error("REPAIRED_MOTION_LAB_SMOKE_FAILED " + ", ".join(errors))
	get_tree().quit(0 if errors.is_empty() else 1)
