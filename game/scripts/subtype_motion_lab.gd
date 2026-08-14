extends Node2D

const CATALOG_PATH := "res://generated/morphology_subtype_lab/v1/catalog.json"
const ASSET_ROOT := "res://generated/morphology_subtype_lab/v1/"
const FORMAT := "nullvector-native-morphology-subtype-runtime-v1"
const LAYERS := ["composite", "semantic", "emission"]
const COLORS := {"humanoid": Color("#37f3ff"), "animalian": Color("#ff4fb7"), "plantlike": Color("#a8ff4f"), "anomaly": Color("#a77bff"), "machine": Color("#ffae37")}
const TEXT := Color("#e9f7ff"); const MUTED := Color("#718ba5"); const RULE := Color("#1d3c5e"); const DEEP := Color("#03070d"); const LIME := Color("#a8ff4f"); const ERROR := Color("#ff526d")

var catalog: Dictionary = {}; var errors: Array[String] = []
var identity_index := 0; var motion_index := 0; var facing_index := 0; var layer_index := 0; var frame := 0
var playing := true; var accumulator := 0.0; var zoom := 7.0; var atlas: ImageTexture
var status_label: Label; var identity_label: Label; var clip_label: Label; var frame_label: Label

func _ready() -> void:
	texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST; _build_ui(); catalog = _load_json(CATALOG_PATH); _validate_catalog()
	if errors.is_empty(): _load_atlas(); _refresh()
	else: status_label.text = "FAIL-CLOSED // " + ", ".join(errors); status_label.modulate = ERROR
	queue_redraw()
	if "--subtype-motion-smoke" in OS.get_cmdline_user_args(): call_deferred("_run_smoke")

func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var file := FileAccess.open(path, FileAccess.READ); if file == null: return {}
	var value = JSON.parse_string(file.get_as_text()); return value if value is Dictionary else {}

func _validate_catalog() -> void:
	if catalog.get("format", "") != FORMAT: errors.append("catalog format")
	if catalog.get("status", "") != "ready" or bool(catalog.get("neural_output", true)): errors.append("authority label")
	var counts: Dictionary = catalog.get("counts", {})
	for pair in [["identity_count", 20], ["clip_count", 400], ["frame_count", 3620], ["atlas_count", 60]]:
		if int(counts.get(pair[0], -1)) != pair[1]: errors.append(str(pair[0]))
	if catalog.get("layers", []) != LAYERS or catalog.get("identities", []).size() != 20: errors.append("registry")
	if errors.is_empty(): status_label.text = "20 CHASSIS ONLINE // 400 CLIPS // 3,620 FRAMES // PROCEDURAL REFERENCE"; status_label.modulate = LIME

func _label(parent: Node, position: Vector2, size: Vector2, text: String, color := TEXT, font := 10) -> Label:
	var label := Label.new(); label.position = position; label.size = size; label.text = text; label.add_theme_font_size_override("font_size", font); label.add_theme_color_override("font_color", color); parent.add_child(label); return label

func _panel(parent: Node, rect: Rect2) -> Panel:
	var panel := Panel.new(); panel.position = rect.position; panel.size = rect.size; var style := StyleBoxFlat.new(); style.bg_color = Color(0.018, 0.045, 0.085, 0.96); style.border_color = RULE; style.set_border_width_all(1); panel.add_theme_stylebox_override("panel", style); parent.add_child(panel); return panel

func _button(parent: Node, position: Vector2, size: Vector2, text: String, callback: Callable) -> void:
	var button := Button.new(); button.position = position; button.size = size; button.text = text; button.focus_mode = Control.FOCUS_NONE; button.add_theme_font_size_override("font_size", 9); button.pressed.connect(callback); parent.add_child(button)

func _build_ui() -> void:
	var canvas := CanvasLayer.new(); add_child(canvas)
	_label(canvas, Vector2(24, 12), Vector2(720, 32), "NULLVECTOR // 20-CHASSIS SUBTYPE MOTION LAB", TEXT, 22)
	_label(canvas, Vector2(26, 45), Vector2(900, 20), "SOFT BILATERAL FORMS // GRAPH-DRIVEN PIXEL MOTION // PROCEDURAL REFERENCE", Color("#37f3ff"), 10)
	status_label = _label(canvas, Vector2(660, 18), Vector2(590, 30), "LOADING", LIME, 9); status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	var left := _panel(canvas, Rect2(18, 80, 330, 610)); _label(left, Vector2(14, 12), Vector2(290, 20), "01 // CHASSIS", Color("#37f3ff"), 11)
	_button(left, Vector2(14, 42), Vector2(60, 30), "Q <", func(): _change_identity(-1)); _button(left, Vector2(80, 42), Vector2(60, 30), "E >", func(): _change_identity(1)); _button(left, Vector2(148, 42), Vector2(74, 30), "- FAMILY", func(): _change_identity(-4)); _button(left, Vector2(228, 42), Vector2(86, 30), "+ FAMILY", func(): _change_identity(4))
	identity_label = _label(left, Vector2(14, 84), Vector2(300, 95), "IDENTITY", TEXT, 11)
	_label(left, Vector2(14, 190), Vector2(290, 20), "02 // MOTION / FACING", Color("#37f3ff"), 11)
	_button(left, Vector2(14, 220), Vector2(55, 30), "W <", func(): _change_motion(-1)); _button(left, Vector2(75, 220), Vector2(55, 30), "S >", func(): _change_motion(1)); _button(left, Vector2(136, 220), Vector2(55, 30), "A <", func(): _change_facing(-1)); _button(left, Vector2(197, 220), Vector2(55, 30), "D >", func(): _change_facing(1)); _button(left, Vector2(258, 220), Vector2(56, 30), "SPACE", func(): playing = not playing)
	clip_label = _label(left, Vector2(14, 264), Vector2(300, 105), "CLIP", TEXT, 10)
	_label(left, Vector2(14, 380), Vector2(290, 20), "03 // FIELD VIEW", Color("#37f3ff"), 11)
	_button(left, Vector2(14, 410), Vector2(60, 30), "Z <", func(): _change_layer(-1)); _button(left, Vector2(80, 410), Vector2(60, 30), "X >", func(): _change_layer(1)); _button(left, Vector2(148, 410), Vector2(74, 30), "1X", func(): zoom = 1.0; queue_redraw()); _button(left, Vector2(228, 410), Vector2(86, 30), "7X", func(): zoom = 7.0; queue_redraw())
	_label(left, Vector2(14, 458), Vector2(300, 110), "COMPOSITE / SEMANTIC / EMISSION\nNATIVE 48PX CELLS\nLOOPS OMIT TERMINAL PROOF FRAME\nNO PYTHON AT RUNTIME", MUTED, 9)
	var stage := _panel(canvas, Rect2(364, 80, 898, 610)); _label(stage, Vector2(14, 12), Vector2(500, 20), "LIVE 768x576 ATLAS READER", Color("#37f3ff"), 10)
	frame_label = _label(stage, Vector2(580, 12), Vector2(290, 20), "FRAME", MUTED, 9); frame_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT

func _identity() -> Dictionary: return catalog.get("identities", [])[posmod(identity_index, 20)]
func _motion() -> String: return str(catalog.get("motions", [])[posmod(motion_index, 13)])
func _facing() -> String: return str(catalog.get("facings", [])[posmod(facing_index, 8)]) if _motion() == "locomote" else "north"
func _clip() -> Dictionary:
	for candidate in _identity().get("clips", []):
		if candidate.get("motion", "") == _motion() and candidate.get("facing", "") == _facing(): return candidate
	return {}

func _load_atlas() -> void:
	atlas = null; var artifact: Dictionary = _identity().get("atlases", {}).get(LAYERS[layer_index], {}); var path := ASSET_ROOT + str(artifact.get("path", ""))
	if not FileAccess.file_exists(path) or FileAccess.get_file_as_bytes(path).size() != int(artifact.get("bytes", -1)) or FileAccess.get_sha256(path) != str(artifact.get("sha256", "")): errors.append("atlas integrity"); return
	var image := Image.load_from_file(path); if image == null or image.get_width() != 768 or image.get_height() != 576: errors.append("atlas shape"); return
	atlas = ImageTexture.create_from_image(image)

func _change_identity(delta: int) -> void: identity_index = posmod(identity_index + delta, 20); frame = 0; _load_atlas(); _refresh(); queue_redraw()
func _change_motion(delta: int) -> void: motion_index = posmod(motion_index + delta, 13); frame = 0; _refresh(); queue_redraw()
func _change_facing(delta: int) -> void: facing_index = posmod(facing_index + delta, 8); frame = 0; _refresh(); queue_redraw()
func _change_layer(delta: int) -> void: layer_index = posmod(layer_index + delta, LAYERS.size()); _load_atlas(); _refresh(); queue_redraw()

func _refresh() -> void:
	if catalog.is_empty(): return
	var identity := _identity(); var clip := _clip(); var family := str(identity.get("family", "?"))
	identity_label.text = "#%02d // %s\n%s\nSEED %s" % [int(identity.get("subtype_id", -1)), family.to_upper(), str(identity.get("subtype", "?")).to_upper(), str(identity.get("seed", "?"))]; identity_label.modulate = COLORS.get(family, TEXT)
	clip_label.text = "%s // %s\n%d FPS // %d STORED\n%s\nFIELD // %s" % [_motion().to_upper(), _facing().to_upper(), int(clip.get("fps", 0)), int(clip.get("frame_count", 0)), "LOOP" if clip.get("loop", false) else "ONE SHOT", LAYERS[layer_index].to_upper()]
	frame_label.text = "FRAME %d / %d // CELL %d" % [frame + 1, int(clip.get("frame_count", 0)), int(clip.get("start_cell", 0)) + frame]

func _process(delta: float) -> void:
	if not errors.is_empty() or not playing: return
	var clip := _clip(); var stored := int(clip.get("frame_count", 1)); var playable := maxi(1, stored - 1 if clip.get("loop", false) else stored); accumulator += delta; var step := 1.0 / maxf(1.0, float(clip.get("fps", 8)))
	while accumulator >= step:
		accumulator -= step
		if frame + 1 < playable: frame += 1
		elif clip.get("loop", false): frame = 0
		else: playing = false
		_refresh(); queue_redraw()

func _draw() -> void:
	draw_rect(Rect2(0, 0, 1280, 720), DEEP); draw_rect(Rect2(365, 81, 896, 608), Color("#03080f"))
	for x in range(390, 1240, 24): draw_line(Vector2(x, 112), Vector2(x, 670), Color(0.12, 0.3, 0.4, 0.14), 1)
	for y in range(112, 671, 24): draw_line(Vector2(390, y), Vector2(1240, y), Color(0.12, 0.3, 0.4, 0.14), 1)
	if atlas == null or catalog.is_empty(): return
	var clip := _clip(); var cell := int(clip.get("start_cell", 0)) + frame; var source := Rect2((cell % 16) * 48, (cell / 16) * 48, 48, 48); var size := Vector2(48, 48) * zoom; var destination := Rect2(Vector2(815, 390) - size * 0.5, size)
	var color: Color = COLORS.get(str(_identity().get("family", "")), Color.WHITE); var glow := color; glow.a = 0.04; draw_circle(Vector2(815, 520), 190, glow); draw_texture_rect_region(atlas, destination, source)

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
			KEY_SPACE: playing = not playing

func _run_smoke() -> void:
	var smoke_errors := errors.duplicate(); var clips := 0; var frames := 0; var atlases := 0
	for identity in catalog.get("identities", []):
		clips += identity.get("clips", []).size(); frames += int(identity.get("frame_count", 0))
		for layer in LAYERS:
			var artifact: Dictionary = identity.get("atlases", {}).get(layer, {}); var path := ASSET_ROOT + str(artifact.get("path", ""))
			if not FileAccess.file_exists(path) or FileAccess.get_sha256(path) != str(artifact.get("sha256", "")): smoke_errors.append("atlas hash")
			else:
				var image := Image.load_from_file(path); if image == null or image.get_width() != 768 or image.get_height() != 576: smoke_errors.append("atlas dimensions")
			atlases += 1
	if clips != 400 or frames != 3620 or atlases != 60: smoke_errors.append("totals")
	var report := {"format": "nullvector-subtype-motion-godot-smoke-v1", "passed": smoke_errors.is_empty(), "identity_count": 20, "clip_count": clips, "frame_count": frames, "atlas_count": atlases, "errors": smoke_errors, "catalog_semantic_sha256": catalog.get("semantic_sha256", "")}
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--subtype-motion-report="):
			var path := argument.trim_prefix("--subtype-motion-report="); var file := FileAccess.open(path, FileAccess.WRITE); if file: file.store_string(JSON.stringify(report, "  ", false)); file.close()
	if smoke_errors.is_empty(): print("SUBTYPE_MOTION_SMOKE_OK identities=20 clips=400 frames=3620 atlases=60")
	else: push_error("SUBTYPE_MOTION_SMOKE_FAILED " + ", ".join(smoke_errors))
	get_tree().quit(0 if smoke_errors.is_empty() else 1)
