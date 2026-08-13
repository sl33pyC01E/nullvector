extends Node2D

const NeuralSpriteScene = preload("res://scripts/neural_sprite.gd")

var registry: Dictionary
var sprites: Array = []
var player: NeuralSprite
var player_index := 0
var player_velocity := Vector2.ZERO
var title_label: Label
var genome_label: Label
var state_label: Label
var help_label: Label
var model_label: Label
var flash := 0.0


func _ready() -> void:
	get_viewport().set_embedding_subwindows(false)
	registry = _load_registry()
	_build_interface()
	if registry.is_empty():
		state_label.text = "NO TRAINED ATLAS FOUND"
		state_label.modulate = Color("#ff4d77")
		return
	_spawn_gallery()
	queue_redraw()


func _load_registry() -> Dictionary:
	var path := "res://generated/sprite_registry.json"
	if not FileAccess.file_exists(path):
		return {}
	var handle := FileAccess.open(path, FileAccess.READ)
	var payload = JSON.parse_string(handle.get_as_text())
	return payload if payload is Dictionary else {}


func _build_interface() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)

	var top_rule := ColorRect.new()
	top_rule.position = Vector2(32, 29)
	top_rule.size = Vector2(1216, 1)
	top_rule.color = Color("#25304a")
	layer.add_child(top_rule)

	title_label = Label.new()
	title_label.position = Vector2(32, 38)
	title_label.text = "NEURAL // SPRITE FORGE"
	title_label.add_theme_font_size_override("font_size", 28)
	title_label.add_theme_color_override("font_color", Color("#eaf7ff"))
	layer.add_child(title_label)

	model_label = Label.new()
	model_label.position = Vector2(34, 73)
	model_label.text = "ABSORBING-STATE CATEGORICAL DIFFUSION"
	model_label.add_theme_font_size_override("font_size", 10)
	model_label.add_theme_color_override("font_color", Color("#50efff"))
	layer.add_child(model_label)

	state_label = Label.new()
	state_label.position = Vector2(1005, 42)
	state_label.size = Vector2(240, 22)
	state_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	state_label.text = "MODEL ONLINE"
	state_label.add_theme_font_size_override("font_size", 11)
	state_label.add_theme_color_override("font_color", Color("#b8ff59"))
	layer.add_child(state_label)

	genome_label = Label.new()
	genome_label.position = Vector2(34, 642)
	genome_label.add_theme_font_size_override("font_size", 13)
	genome_label.add_theme_color_override("font_color", Color("#d5dced"))
	layer.add_child(genome_label)

	help_label = Label.new()
	help_label.position = Vector2(690, 642)
	help_label.size = Vector2(554, 42)
	help_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	help_label.text = "WASD  MOVE     MOUSE  AIM     LMB / J  ATTACK     R  MUTATE"
	help_label.add_theme_font_size_override("font_size", 10)
	help_label.add_theme_color_override("font_color", Color("#7f8aa5"))
	layer.add_child(help_label)


func _spawn_gallery() -> void:
	var entries: Array = registry.get("sprites", [])
	if entries.is_empty():
		state_label.text = "REGISTRY EMPTY"
		return
	model_label.text = "CATEGORICAL DIFFUSION // " + str(registry.get("model_hash", "UNKNOWN"))
	for index in range(entries.size()):
		var actor := NeuralSpriteScene.new() as NeuralSprite
		actor.configure(entries[index])
		add_child(actor)
		actor.scale = Vector2.ONE * 2.4
		var column := index % 8
		var row := index / 8
		actor.position = Vector2(530 + column * 90, 220 + row * 128)
		actor.modulate.a = 0.72
		actor.set_animation(["idle", "move", "attack"][index % 3], true)
		sprites.append(actor)
	_select_player(0)


func _select_player(next_index: int) -> void:
	if sprites.is_empty():
		return
	if is_instance_valid(player):
		player.queue_free()
	var entries: Array = registry["sprites"]
	player_index = posmod(next_index, entries.size())
	player = NeuralSpriteScene.new() as NeuralSprite
	player.configure(entries[player_index])
	add_child(player)
	player.position = Vector2(245, 360)
	player.scale = Vector2.ONE * 5.2
	player.set_animation("idle", true)
	var manifest: Dictionary = entries[player_index]
	var genome: Dictionary = manifest.get("genome", {})
	genome_label.text = (
		"GENOME 0x%08X    ARCHETYPE %s    LATENT TRAITS %s"
		% [
			int(manifest.get("seed", 0)),
			str(manifest.get("archetype", "unknown")).to_upper(),
			_format_genes(genome.get("genes", [])),
		]
	)


func _format_genes(values: Array) -> String:
	var output: Array[String] = []
	for index in range(min(4, values.size())):
		output.append("%.2f" % float(values[index]))
	return " / ".join(output)


func _process(delta: float) -> void:
	if not is_instance_valid(player):
		return
	var direction := Input.get_vector("move_left", "move_right", "move_up", "move_down")
	var target_velocity := direction * 220.0
	player_velocity = player_velocity.lerp(target_velocity, 1.0 - exp(-delta * 15.0))
	player.position += player_velocity * delta
	player.position.x = clamp(player.position.x, 110.0, 430.0)
	player.position.y = clamp(player.position.y, 175.0, 555.0)
	if direction.length_squared() > 0.04:
		player.set_animation("move")
	elif player.animation_name == "move":
		player.set_animation("idle")

	var aim := get_global_mouse_position() - player.position
	if aim.length_squared() > 1.0:
		player.rotation = aim.angle() + PI * 0.5

	if Input.is_action_just_pressed("attack"):
		player.set_animation("attack", true)
		flash = 1.0
	if Input.is_action_just_pressed("mutate"):
		_select_player(player_index + 1)
		flash = 0.7
	if flash > 0.0:
		flash = maxf(0.0, flash - delta * 3.5)
		queue_redraw()


func _draw() -> void:
	draw_rect(Rect2(0, 0, 1280, 720), Color("#03050b"))
	for x in range(32, 1280, 32):
		var alpha := 0.075 if x % 128 == 0 else 0.026
		draw_line(Vector2(x, 105), Vector2(x, 620), Color(0.22, 0.55, 0.72, alpha), 1.0)
	for y in range(108, 621, 32):
		var alpha := 0.075 if y % 128 == 12 else 0.026
		draw_line(Vector2(0, y), Vector2(1280, y), Color(0.22, 0.55, 0.72, alpha), 1.0)
	draw_line(Vector2(480, 122), Vector2(480, 600), Color("#26314a"), 1.0)
	draw_rect(Rect2(82, 140, 330, 438), Color(0.05, 0.08, 0.14, 0.32), true)
	draw_rect(Rect2(82, 140, 330, 438), Color("#26314a"), false, 1.0)
	draw_string(
		ThemeDB.fallback_font,
		Vector2(100, 169),
		"LIVE RIG / PLAYER CANDIDATE",
		HORIZONTAL_ALIGNMENT_LEFT,
		-1,
		11,
		Color("#8290aa")
	)
	if flash > 0.0:
		draw_circle(Vector2(245, 360), 62.0 + flash * 34.0, Color(0.3, 0.95, 1.0, flash * 0.08), false, 3.0)
