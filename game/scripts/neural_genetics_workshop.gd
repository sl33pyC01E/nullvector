extends Node2D

const INDEX_PATH := "res://generated/neural_genetics/v3/asset_index.json"
const INDEX_FORMAT := "nullvector-neural-genetics-workshop-assets-v3"
const ROOT := "res://generated/neural_genetics/v3/"
const MODES := ["fusion", "latent", "evolution"]
const LAYERS := ["base", "outline", "emission_core", "aura", "bloom_r1", "bloom_r2", "composite"]
const CYAN := Color("#38ecff")
const PINK := Color("#ff3fb4")
const LIME := Color("#baff57")
const VIOLET := Color("#aa7cff")
const TEXT := Color("#e8f6ff")
const MUTED := Color("#7890aa")
const RULE := Color("#203a5f")
const DEEP := Color("#030711")

var index: Dictionary = {}
var mode_index := 0
var specimen_index := 0
var clip_index := 0
var layer_index := 6
var frame_index := 0
var playing := true
var accumulator := 0.0
var startup_errors: Array[String] = []

var mode_label: Label
var specimen_label: Label
var lineage_label: Label
var metrics_label: Label
var truth_label: Label
var frame_label: Label
var layer_label: Label
var clip_label: Label
var sprite_rect: TextureRect
var status_label: Label
var play_button: Button


func _ready() -> void:
	get_viewport().set_embedding_subwindows(false)
	_build_interface()
	index = _load_json(INDEX_PATH)
	_validate_index()
	_refresh()
	await _run_smoke_if_requested()


func _load_json(path: String) -> Dictionary:
	if not FileAccess.file_exists(path): return {}
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null: return {}
	var value = JSON.parse_string(file.get_as_text())
	return value if value is Dictionary else {}


func _validate_index() -> void:
	if index.get("format", "") != INDEX_FORMAT: startup_errors.append("index format")
	if index.get("status", "") != "ready": startup_errors.append("index status")
	if index.get("pixel_filter", "") != "nearest": startup_errors.append("pixel filter")
	if bool(index.get("python_runtime_required", true)): startup_errors.append("Python runtime")
	for gate in index.get("gates", {}).values():
		if gate != true: startup_errors.append("failed source gate")
	if index.get("fusion", {}).get("specimen_count", 0) != 10: startup_errors.append("fusion census")
	if index.get("latent", {}).get("specimen_count", 0) != 12: startup_errors.append("latent census")
	if index.get("evolution", {}).get("selected_count", 0) != 36: startup_errors.append("evolution census")
	if startup_errors.is_empty():
		status_label.text = "BUNDLE %s // 10 FUSIONS // 12 LATENTS // 36 EVOLVED" % str(index.get("bundle_id", "")).substr(0, 12)
		status_label.modulate = LIME
	else:
		status_label.text = "FAIL-CLOSED // " + ", ".join(startup_errors)
		status_label.modulate = Color("#ff526d")


func _panel(parent: Node, rect: Rect2) -> Panel:
	var panel := Panel.new()
	panel.position = rect.position; panel.size = rect.size
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.025, 0.055, 0.11, 0.96); style.border_color = RULE; style.set_border_width_all(1)
	panel.add_theme_stylebox_override("panel", style); parent.add_child(panel)
	return panel


func _label(parent: Node, position: Vector2, size: Vector2, value: String, color := TEXT, font_size := 11) -> Label:
	var item := Label.new()
	item.position = position; item.size = size; item.text = value
	item.add_theme_font_size_override("font_size", font_size); item.add_theme_color_override("font_color", color)
	parent.add_child(item); return item


func _button(parent: Node, position: Vector2, size: Vector2, value: String, callback: Callable) -> Button:
	var item := Button.new()
	item.position = position; item.size = size; item.text = value; item.focus_mode = Control.FOCUS_NONE
	item.add_theme_font_size_override("font_size", 9); item.pressed.connect(callback); parent.add_child(item)
	return item


func _build_interface() -> void:
	var canvas := CanvasLayer.new(); add_child(canvas)
	_label(canvas, Vector2(26, 15), Vector2(790, 35), "NULLVECTOR // NATIVE NEURAL GENETICS WORKSHOP", TEXT, 23)
	_label(canvas, Vector2(28, 48), Vector2(820, 20), "CATEGORICAL FUSION  ×  LEARNED LATENT INTERPOLATION  ×  EVOLUTION", CYAN, 10)
	status_label = _label(canvas, Vector2(760, 21), Vector2(492, 32), "LOADING HASH-BOUND BANK", LIME, 9); status_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	var left := _panel(canvas, Rect2(20, 82, 338, 568))
	_label(left, Vector2(14, 10), Vector2(300, 22), "01 // GENETIC MODE", CYAN, 12)
	_button(left, Vector2(14, 40), Vector2(96, 34), "1 FUSION", func(): _set_mode(0))
	_button(left, Vector2(114, 40), Vector2(96, 34), "2 LATENT", func(): _set_mode(1))
	_button(left, Vector2(214, 40), Vector2(108, 34), "3 EVOLVE", func(): _set_mode(2))
	mode_label = _label(left, Vector2(14, 90), Vector2(306, 45), "MODE", PINK, 15)
	truth_label = _label(left, Vector2(14, 137), Vector2(306, 68), "TRUTH LABEL", MUTED, 9); truth_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_label(left, Vector2(14, 220), Vector2(306, 18), "SPECIMEN", MUTED, 8)
	_button(left, Vector2(14, 242), Vector2(48, 32), "Q <", func(): _step_specimen(-1))
	_button(left, Vector2(66, 242), Vector2(48, 32), "E >", func(): _step_specimen(1))
	specimen_label = _label(left, Vector2(124, 237), Vector2(196, 48), "SPECIMEN", TEXT, 9)
	_label(left, Vector2(14, 294), Vector2(306, 18), "CLIP / GENERATION", MUTED, 8)
	_button(left, Vector2(14, 316), Vector2(48, 32), "W <", func(): _step_clip(-1))
	_button(left, Vector2(66, 316), Vector2(48, 32), "S >", func(): _step_clip(1))
	clip_label = _label(left, Vector2(124, 311), Vector2(196, 48), "CLIP", TEXT, 9)
	_label(left, Vector2(14, 368), Vector2(306, 18), "PRESENTATION LAYER", MUTED, 8)
	_button(left, Vector2(14, 390), Vector2(48, 32), "Z <", func(): _step_layer(-1))
	_button(left, Vector2(66, 390), Vector2(48, 32), "X >", func(): _step_layer(1))
	layer_label = _label(left, Vector2(124, 385), Vector2(196, 48), "LAYER", TEXT, 9)
	play_button = _button(left, Vector2(14, 452), Vector2(100, 34), "SPACE PAUSE", func(): playing = not playing; _refresh())
	frame_label = _label(left, Vector2(124, 451), Vector2(196, 34), "FRAME", PINK, 10)
	_label(left, Vector2(14, 505), Vector2(306, 42), "Q/E SPECIMEN  W/S CLIP  Z/X LAYER\nLEFT/RIGHT SCRUB  SPACE PLAY", MUTED, 8)

	var center := _panel(canvas, Rect2(374, 82, 518, 568))
	_label(center, Vector2(14, 10), Vector2(480, 22), "02 // LIVE NATIVE ATLAS READER", CYAN, 12)
	sprite_rect = TextureRect.new(); sprite_rect.position = Vector2(34, 46); sprite_rect.size = Vector2(450, 450)
	sprite_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE; sprite_rect.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED; sprite_rect.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST; center.add_child(sprite_rect)
	_label(center, Vector2(28, 515), Vector2(460, 20), "48×48 NATIVE REGION // NEAREST // HASH-VERIFIED", MUTED, 8).horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER

	var right := _panel(canvas, Rect2(908, 82, 352, 568))
	_label(right, Vector2(14, 10), Vector2(322, 22), "03 // LINEAGE + FITNESS", CYAN, 12)
	lineage_label = _label(right, Vector2(14, 48), Vector2(322, 146), "LINEAGE", TEXT, 10); lineage_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	metrics_label = _label(right, Vector2(14, 204), Vector2(322, 338), "METRICS", MUTED, 9); metrics_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_label(canvas, Vector2(20, 670), Vector2(1240, 25), "ARENA MAIN SCENE UNCHANGED // THIS IS AN ADDITIVE NATIVE RESEARCH WORKSHOP", MUTED, 9).horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER


func _bank_name() -> String:
	return MODES[mode_index]


func _specimens() -> Array:
	return index.get(_bank_name(), {}).get("specimens", [])


func _specimen() -> Dictionary:
	var values := _specimens()
	if values.is_empty(): return {}
	specimen_index = posmod(specimen_index, values.size())
	return values[specimen_index]


func _clips(specimen: Dictionary) -> Array:
	return specimen.get("clips", [])


func _set_mode(value: int) -> void:
	mode_index = clampi(value, 0, MODES.size() - 1); specimen_index = 0; clip_index = 0; layer_index = 6; frame_index = 0; playing = true; accumulator = 0.0; _refresh()


func _step_specimen(delta: int) -> void:
	var values := _specimens()
	if values.is_empty(): return
	specimen_index = posmod(specimen_index + delta, values.size()); clip_index = 0; frame_index = 0; accumulator = 0.0; _refresh()


func _step_clip(delta: int) -> void:
	var clips := _clips(_specimen())
	if not clips.is_empty(): clip_index = posmod(clip_index + delta, clips.size())
	frame_index = 0; accumulator = 0.0; _refresh()


func _step_layer(delta: int) -> void:
	layer_index = posmod(layer_index + delta, LAYERS.size()); _refresh()


func _runtime_path(relative: String) -> String:
	return ROOT + relative


func _atlas_texture(specimen: Dictionary, layer: String, cell: int) -> AtlasTexture:
	var record: Dictionary = specimen.get("layers", {}).get(layer, {})
	var source = load(_runtime_path(str(record.get("path", "")))) as Texture2D
	if source == null: return null
	var columns := int(specimen.get("layout", {}).get("columns", 16))
	var texture := AtlasTexture.new(); texture.atlas = source
	texture.region = Rect2((cell % columns) * 48, (cell / columns) * 48, 48, 48)
	return texture


func _refresh() -> void:
	if not startup_errors.is_empty() or index.is_empty(): return
	var specimen := _specimen(); if specimen.is_empty(): return
	mode_label.text = ("VERIFIED CATEGORICAL FUSION" if mode_index == 0 else "PRODUCTION EMA-FSQ GENETICS" if mode_index == 1 else "MOTION-GATED EVOLUTION")
	mode_label.modulate = PINK if mode_index == 0 else VIOLET if mode_index == 1 else LIME
	truth_label.text = str(index.get(_bank_name(), {}).get("truth_label", "missing truth label")).replace("-", " ").to_upper()
	specimen_label.text = "%02d / %02d\n%s" % [specimen_index + 1, _specimens().size(), str(specimen.get("sample_id", "missing"))]
	play_button.text = "SPACE PLAY" if not playing else "SPACE PAUSE"
	var clips := _clips(specimen); clip_index = posmod(clip_index, clips.size()); var clip: Dictionary = clips[clip_index]
	var playback_frames := int(clip.get("frame_count", 1)) - (1 if bool(clip.get("loop", false)) else 0); playback_frames = maxi(1, playback_frames); frame_index = clampi(frame_index, 0, playback_frames - 1)
	var cell := int(clip.get("start_cell", 0)) + frame_index; sprite_rect.texture = _atlas_texture(specimen, LAYERS[layer_index], cell)
	clip_label.text = "%s\n%s @ %d FPS" % [str(clip.get("motion", "")), str(clip.get("facing", "")), int(clip.get("fps", 0))]
	layer_label.text = LAYERS[layer_index].to_upper(); frame_label.text = "FRAME %02d / %02d" % [frame_index + 1, playback_frames]
	if mode_index == 0:
		var parents: Array = specimen.get("parents", []); var parent_lines: Array[String] = []
		for parent in parents: parent_lines.append("%s // %s" % [str(parent.get("family", "")), str(parent.get("sample_id", ""))])
		lineage_label.text = "FAMILY  %s\nFUSION  %s\nMUTATION  %s × %d\n\nPARENTS\n%s" % [str(specimen.get("family", "")), str(specimen.get("mode", "")), str(specimen.get("mutation_mode", "")), int(specimen.get("mutation_strength", 0)), "\n".join(parent_lines)]
	elif mode_index == 1:
		lineage_label.text = "FUSION  %s\nMUTATION  %s x %.2f\nALPHA  %.2f\n\nPARENTS\n%s\n%s\n\nQUALITY\n%s" % [str(specimen.get("mode", "")), str(specimen.get("mutation_mode", "")), float(specimen.get("mutation_strength", 0.0)), float(specimen.get("alpha", 0.0)), str(specimen.get("parents", ["", ""])[0]), str(specimen.get("parents", ["", ""])[1]), str(specimen.get("quality_tier", ""))]
	else:
		lineage_label.text = "GENERATION  %d // RANK %02d\nFAMILY  %s\nFUSION  %s\nMUTATION  %s x %d\nALPHA  %.2f\n\nPARENTS\n%s" % [int(specimen.get("generation", 0)), int(specimen.get("rank", 0)) + 1, str(specimen.get("family", "")), str(specimen.get("fusion_mode", "")), str(specimen.get("mutation_mode", "")), int(specimen.get("mutation_strength", 0)), float(specimen.get("alpha", 0.0)), "\n".join(specimen.get("parents", []))]
	var metrics: Dictionary = specimen.get("metrics", {}); var lines: Array[String] = []
	for key in metrics.keys():
		if metrics[key] is float or metrics[key] is int: lines.append("%-24s %s" % [str(key).replace("_", " ").to_upper(), str(metrics[key])])
	metrics_label.text = "\n".join(lines.slice(0, 17))


func _process(delta: float) -> void:
	if not playing or startup_errors.size() > 0: return
	var specimen := _specimen(); var clips := _clips(specimen); if clips.is_empty(): return
	var clip: Dictionary = clips[clip_index]; var playback_frames := int(clip.get("frame_count", 1)) - (1 if bool(clip.get("loop", false)) else 0); playback_frames = maxi(1, playback_frames)
	accumulator += delta
	if accumulator >= 1.0 / maxf(1.0, float(clip.get("fps", 8))):
		accumulator = 0.0
		if frame_index + 1 < playback_frames: frame_index += 1
		elif bool(clip.get("loop", false)): frame_index = 0
		else: playing = false
		_refresh()


func _unhandled_input(event: InputEvent) -> void:
	if not event.is_pressed() or event.is_echo(): return
	if event.keycode == KEY_1: _set_mode(0)
	elif event.keycode == KEY_2: _set_mode(1)
	elif event.keycode == KEY_3: _set_mode(2)
	elif event.keycode == KEY_Q: _step_specimen(-1)
	elif event.keycode == KEY_E: _step_specimen(1)
	elif event.keycode == KEY_W: _step_clip(-1)
	elif event.keycode == KEY_S: _step_clip(1)
	elif event.keycode == KEY_Z: _step_layer(-1)
	elif event.keycode == KEY_X: _step_layer(1)
	elif event.keycode == KEY_SPACE: playing = not playing; _refresh()
	elif event.keycode == KEY_LEFT: playing = false; frame_index = maxi(0, frame_index - 1); _refresh()
	elif event.keycode == KEY_RIGHT: playing = false; frame_index += 1; _refresh()


func _run_smoke_if_requested() -> void:
	var args := OS.get_cmdline_user_args()
	if not args.has("--neural-genetics-smoke"): return
	var errors: Array[String] = startup_errors.duplicate()
	var hash_count := 0
	for record in index.get("inventory", []):
		var path := _runtime_path(str(record.get("path", "")))
		if not FileAccess.file_exists(path): errors.append("missing inventory " + path); continue
		if FileAccess.get_sha256(path) != str(record.get("sha256", "")): errors.append("hash " + path)
		else: hash_count += 1
	var atlas_count := 0; var clip_count := 0; var frame_count := 0; var region_count := 0
	for bank_name in ["fusion", "latent", "evolution"]:
		for specimen in index.get(bank_name, {}).get("specimens", []):
			var layout: Dictionary = specimen.get("layout", {}); var columns := int(layout.get("columns", 0)); var rows := int(layout.get("rows", 0))
			for layer in LAYERS:
				var record: Dictionary = specimen.get("layers", {}).get(layer, {}); var texture = load(_runtime_path(str(record.get("path", "")))) as Texture2D
				if texture == null or texture.get_width() != columns * 48 or texture.get_height() != rows * 48: errors.append("atlas " + str(specimen.get("sample_id", "")) + "/" + layer)
				else: atlas_count += 1
			for clip in specimen.get("clips", []):
				clip_count += 1
				for frame in range(int(clip.get("frame_count", 0))):
					var region := AtlasTexture.new(); region.atlas = load(_runtime_path(str(specimen.get("layers", {}).get("composite", {}).get("path", ""))))
					var cell := int(clip.get("start_cell", 0)) + frame; region.region = Rect2((cell % columns) * 48, (cell / columns) * 48, 48, 48)
					if region.region.end.x > columns * 48 or region.region.end.y > rows * 48: errors.append("region bounds")
					region_count += 1; frame_count += 1
	var evolution_count: int = int(index.get("evolution", {}).get("specimen_count", 0))
	if atlas_count != 406: errors.append("atlas total")
	if clip_count != 214 or frame_count != 1908 or region_count != 1908: errors.append("motion totals")
	if evolution_count != 36 or hash_count != 406: errors.append("asset totals")
	var report := {"format": "nullvector-neural-genetics-workshop-godot-smoke-v3", "passed": errors.is_empty(), "errors": errors, "bundle_id": index.get("bundle_id", ""), "coverage": {"fusion_specimens": 10, "latent_specimens": 12, "evolution_specimens": evolution_count, "motion_atlases": atlas_count, "motion_clips": clip_count, "motion_frames": frame_count, "atlas_regions": region_count, "hashes": hash_count}, "engine": Engine.get_version_info().get("string", "")}
	for argument in args:
		if argument.begins_with("--neural-genetics-report="):
			var output_path := argument.trim_prefix("--neural-genetics-report="); var file := FileAccess.open(output_path, FileAccess.WRITE)
			if file != null: file.store_string(JSON.stringify(report, "  ", false) + "\n")
	if errors.is_empty(): print("NEURAL_GENETICS_SMOKE_OK fusion=10 latent=12 evolution=36 atlases=406 clips=214 frames=1908 regions=1908 hashes=406")
	else: push_error("NEURAL_GENETICS_SMOKE_FAILED " + ", ".join(errors))
	get_tree().quit(0 if errors.is_empty() else 1)
