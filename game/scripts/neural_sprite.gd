class_name NeuralSprite
extends Node2D

signal animation_event(event_name: String)

var manifest: Dictionary
var animation_name := "idle"
var frame_index := 0
var frame_clock_ms := 0.0
var playback_speed := 1.0
var genome_phase := 0.0

var _base := Sprite2D.new()
var _emission := Sprite2D.new()
var _glow_soft := Sprite2D.new()
var _glow_wide := Sprite2D.new()


func _ready() -> void:
	var additive := CanvasItemMaterial.new()
	additive.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
	for sprite in [_glow_wide, _glow_soft, _emission, _base]:
		sprite.region_enabled = true
		sprite.centered = true
		sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
		add_child(sprite)
	_glow_wide.material = additive
	_glow_soft.material = additive
	_emission.material = additive
	_glow_wide.modulate = Color(0.16, 0.65, 1.0, 0.10)
	_glow_soft.modulate = Color(0.35, 0.92, 1.0, 0.22)
	_emission.modulate = Color(0.72, 0.98, 1.0, 0.76)
	_glow_wide.scale = Vector2(1.34, 1.34)
	_glow_soft.scale = Vector2(1.15, 1.15)
	if not manifest.is_empty():
		_apply_manifest()


func configure(sprite_manifest: Dictionary, generated_root := "res://generated/") -> void:
	manifest = sprite_manifest
	genome_phase = fmod(float(int(manifest.get("seed", 0))) * 0.000173, TAU)
	if is_node_ready():
		_apply_manifest(generated_root)


func _apply_manifest(generated_root := "res://generated/") -> void:
	var atlas_path := generated_root + str(manifest.get("atlas", ""))
	var emission_path := generated_root + str(manifest.get("emission_atlas", ""))
	var atlas_texture = load(atlas_path)
	var emission_texture = load(emission_path)
	_base.texture = atlas_texture
	_emission.texture = emission_texture
	_glow_soft.texture = emission_texture
	_glow_wide.texture = emission_texture
	set_animation("idle", true)


func set_animation(next_animation: String, restart := false) -> void:
	if manifest.is_empty():
		return
	var animations: Dictionary = manifest.get("animations", {})
	if not animations.has(next_animation):
		return
	if next_animation != animation_name or restart:
		animation_name = next_animation
		frame_index = 0
		frame_clock_ms = 0.0
		_apply_frame()


func _process(delta: float) -> void:
	if manifest.is_empty():
		return
	var animations: Dictionary = manifest.get("animations", {})
	var animation: Dictionary = animations.get(animation_name, {})
	var frames: Array = animation.get("frames", [])
	if frames.is_empty():
		return
	frame_clock_ms += delta * 1000.0 * playback_speed
	var current: Dictionary = frames[frame_index]
	var duration := float(current.get("duration_ms", 100))
	if frame_clock_ms >= duration:
		frame_clock_ms -= duration
		frame_index += 1
		if frame_index >= frames.size():
			if bool(animation.get("loop", false)):
				frame_index = 0
			else:
				set_animation("idle", true)
				return
		_apply_frame()

	var pulse := 0.5 + 0.5 * sin(Time.get_ticks_msec() * 0.006 + genome_phase)
	_emission.modulate.a = 0.62 + pulse * 0.26
	_glow_soft.modulate.a = 0.13 + pulse * 0.11
	_glow_wide.modulate.a = 0.05 + pulse * 0.055


func _apply_frame() -> void:
	var animation: Dictionary = manifest["animations"][animation_name]
	var frames: Array = animation["frames"]
	if frame_index >= frames.size():
		return
	var frame: Dictionary = frames[frame_index]
	var values: Array = frame["rect"]
	var region := Rect2(float(values[0]), float(values[1]), float(values[2]), float(values[3]))
	for sprite in [_base, _emission, _glow_soft, _glow_wide]:
		sprite.region_rect = region
	var event_value = frame.get("event", null)
	if event_value != null:
		animation_event.emit(str(event_value))
