extends Node2D

const NeuralSpriteClass = preload("res://scripts/neural_sprite.gd")

const VIEW_SIZE := Vector2(1280.0, 720.0)
const ARENA_RECT := Rect2(-1200.0, -800.0, 2400.0, 1600.0)
const PLAYER_RADIUS := 22.0
const BOSS_TIME := 180.0
const MAX_ENEMIES := 90

const C_BG := Color("#03050b")
const C_PANEL := Color("#09101de8")
const C_CYAN := Color("#50efff")
const C_BLUE := Color("#4388ff")
const C_MAGENTA := Color("#ff4dce")
const C_LIME := Color("#b8ff59")
const C_RED := Color("#ff4d77")
const C_TEXT := Color("#eaf7ff")
const C_MUTED := Color("#7f8aa5")

enum GameState { TITLE, RUNNING, LEVEL_UP, PAUSED, DEAD, VICTORY }


class EnemyData:
	extends RefCounted
	var sprite: Node2D
	var archetype := "dart"
	var pos := Vector2.ZERO
	var velocity := Vector2.ZERO
	var charge_direction := Vector2.ZERO
	var hp := 10.0
	var max_hp := 10.0
	var speed := 100.0
	var radius := 20.0
	var damage := 8.0
	var xp_value := 1
	var attack_cooldown := 1.0
	var contact_cooldown := 0.0
	var phase := 0.0
	var mode := 0
	var flash := 0.0
	var dash_stamp := -1
	var dead := false
	var is_boss := false


class ProjectileData:
	extends RefCounted
	var pos := Vector2.ZERO
	var previous := Vector2.ZERO
	var velocity := Vector2.ZERO
	var damage := 1.0
	var radius := 4.0
	var life := 1.0
	var friendly := true
	var pierce := 0
	var color := Color.WHITE
	var hit_ids: Dictionary = {}


class PickupData:
	extends RefCounted
	var pos := Vector2.ZERO
	var velocity := Vector2.ZERO
	var value := 1
	var life := 18.0
	var phase := 0.0


class ParticleData:
	extends RefCounted
	var pos := Vector2.ZERO
	var velocity := Vector2.ZERO
	var life := 1.0
	var max_life := 1.0
	var size := 2.0
	var color := Color.WHITE
	var streak := false


class FloaterData:
	extends RefCounted
	var pos := Vector2.ZERO
	var text := ""
	var life := 0.7
	var max_life := 0.7
	var color := Color.WHITE


var rng := RandomNumberGenerator.new()
var state := GameState.TITLE
var registry: Dictionary = {}
var manifests_by_archetype: Dictionary = {}
var manifest_cursor: Dictionary = {}

var camera: Camera2D
var player_sprite: Node2D
var title_sprites: Array = []
var title_origins: Array[Vector2] = []
var enemies: Array = []
var projectiles: Array = []
var pickups: Array = []
var particles: Array = []
var floaters: Array = []

var player_pos := Vector2.ZERO
var player_velocity := Vector2.ZERO
var aim_direction := Vector2.UP
var hp := 100.0
var max_hp := 100.0
var move_speed := 250.0
var fire_interval := 0.18
var fire_cooldown := 0.0
var bullet_damage := 16.0
var bullet_speed := 820.0
var bullet_radius := 4.0
var bullet_pierce := 0
var projectile_count := 1
var crit_chance := 0.05
var pickup_radius := 115.0
var regeneration := 0.0
var dash_cooldown_max := 1.65
var dash_cooldown := 0.0
var dash_time := 0.0
var dash_serial := 0
var dash_damage := 38.0
var invulnerable := 0.0

var elapsed := 0.0
var spawn_clock := 0.0
var kills := 0
var score := 0
var level := 1
var xp := 0
var xp_needed := 12
var pending_levels := 0
var boss_spawned := false
var boss_ref: EnemyData
var run_seed := 0
var announcement_time := 0.0
var announcement_text := ""
var camera_shake := 0.0
var upgrade_levels: Dictionary = {}
var offered_upgrades: Array[String] = []

var ui_layer: CanvasLayer
var hud: Control
var title_overlay: Control
var pause_overlay: Control
var level_overlay: Control
var outcome_overlay: Control
var hp_bar: ProgressBar
var xp_bar: ProgressBar
var boss_bar: ProgressBar
var hp_label: Label
var level_label: Label
var timer_label: Label
var score_label: Label
var director_label: Label
var dash_label: Label
var boss_label: Label
var announcement_label: Label
var outcome_title: Label
var outcome_stats: Label
var level_title: Label
var upgrade_buttons: Array[Button] = []

var upgrade_catalog := [
	{
		"id": "overclock",
		"name": "SYNAPTIC OVERCLOCK",
		"description": "+18% fire rate",
		"max": 6,
	},
	{
		"id": "rail_coils",
		"name": "RAIL COILS",
		"description": "+28% projectile damage",
		"max": 6,
	},
	{
		"id": "forked_signal",
		"name": "FORKED SIGNAL",
		"description": "+1 projectile, wider spread",
		"max": 4,
	},
	{
		"id": "phase_bore",
		"name": "PHASE BORE",
		"description": "+1 enemy pierced",
		"max": 4,
	},
	{
		"id": "flux_drive",
		"name": "FLUX DRIVE",
		"description": "+14% movement speed",
		"max": 5,
	},
	{
		"id": "capacitor",
		"name": "LIVING CAPACITOR",
		"description": "+25 max integrity and repair 25",
		"max": 5,
	},
	{
		"id": "magnetism",
		"name": "GRAVITIC MAGNET",
		"description": "+65 pickup attraction radius",
		"max": 5,
	},
	{
		"id": "phase_lens",
		"name": "PHASE LENS",
		"description": "+16% bolt speed and +1 bolt radius",
		"max": 5,
	},
	{
		"id": "nanoshield",
		"name": "NANO RECURSION",
		"description": "+0.7 integrity regenerated / second",
		"max": 5,
	},
	{
		"id": "criticality",
		"name": "CRITICALITY ENGINE",
		"description": "+9% critical hit chance",
		"max": 5,
	},
	{
		"id": "nova_dash",
		"name": "NOVA DASH",
		"description": "+45% dash damage, -12% dash cooldown",
		"max": 5,
	},
]


func _ready() -> void:
	process_mode = Node.PROCESS_MODE_ALWAYS
	rng.randomize()
	registry = _load_registry()
	_index_manifests()
	_build_camera()
	_build_interface()
	_spawn_title_cast()
	_update_hud()
	queue_redraw()
	if "--arena-smoke" in OS.get_cmdline_user_args():
		call_deferred("_start_smoke_test")


func _load_registry() -> Dictionary:
	var path := "res://generated/sprite_registry.json"
	if not FileAccess.file_exists(path):
		push_error("Neural sprite registry is missing: " + path)
		return {}
	var handle := FileAccess.open(path, FileAccess.READ)
	var payload = JSON.parse_string(handle.get_as_text())
	if payload is Dictionary:
		return payload
	push_error("Neural sprite registry is invalid JSON")
	return {}


func _index_manifests() -> void:
	manifests_by_archetype.clear()
	for entry in registry.get("sprites", []):
		var archetype := str(entry.get("archetype", "unknown"))
		if not manifests_by_archetype.has(archetype):
			manifests_by_archetype[archetype] = []
		manifests_by_archetype[archetype].append(entry)
		manifest_cursor[archetype] = 0


func _manifest_for(archetype: String) -> Dictionary:
	var entries: Array = manifests_by_archetype.get(archetype, [])
	if entries.is_empty():
		var all_entries: Array = registry.get("sprites", [])
		return all_entries[0] if not all_entries.is_empty() else {}
	var cursor := int(manifest_cursor.get(archetype, 0))
	manifest_cursor[archetype] = cursor + 1
	return entries[posmod(cursor, entries.size())]


func _build_camera() -> void:
	camera = Camera2D.new()
	camera.enabled = true
	camera.position = Vector2.ZERO
	camera.position_smoothing_enabled = true
	camera.position_smoothing_speed = 9.0
	add_child(camera)


func _make_label(
	parent: Node,
	text_value: String,
	position_value: Vector2,
	size_value: Vector2,
	font_size: int,
	color: Color = C_TEXT,
	alignment := HORIZONTAL_ALIGNMENT_LEFT
) -> Label:
	var label := Label.new()
	label.text = text_value
	label.position = position_value
	label.size = size_value
	label.horizontal_alignment = alignment
	label.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	label.add_theme_font_size_override("font_size", font_size)
	label.add_theme_color_override("font_color", color)
	parent.add_child(label)
	return label


func _panel_style(color: Color, border_color: Color = Color.TRANSPARENT, width := 0) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = color
	style.border_color = border_color
	style.set_border_width_all(width)
	style.corner_radius_top_left = 3
	style.corner_radius_top_right = 3
	style.corner_radius_bottom_left = 3
	style.corner_radius_bottom_right = 3
	return style


func _style_button(button: Button, accent := C_CYAN) -> void:
	button.add_theme_font_size_override("font_size", 16)
	button.add_theme_color_override("font_color", C_TEXT)
	button.add_theme_color_override("font_hover_color", Color.WHITE)
	button.add_theme_color_override("font_pressed_color", C_BG)
	button.add_theme_stylebox_override("normal", _panel_style(Color("#0b1725ed"), Color("#263c58"), 1))
	button.add_theme_stylebox_override("hover", _panel_style(Color(accent, 0.18), accent, 2))
	button.add_theme_stylebox_override("pressed", _panel_style(accent, accent, 1))
	button.add_theme_stylebox_override("focus", _panel_style(Color(accent, 0.10), accent, 2))


func _make_overlay() -> Control:
	var overlay := Control.new()
	overlay.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	var shade := ColorRect.new()
	shade.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	shade.color = Color("#02040ad9")
	shade.mouse_filter = Control.MOUSE_FILTER_STOP
	overlay.add_child(shade)
	ui_layer.add_child(overlay)
	return overlay


func _build_interface() -> void:
	ui_layer = CanvasLayer.new()
	ui_layer.layer = 10
	add_child(ui_layer)

	hud = Control.new()
	hud.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	hud.mouse_filter = Control.MOUSE_FILTER_IGNORE
	ui_layer.add_child(hud)

	var top_back := ColorRect.new()
	top_back.position = Vector2(0, 0)
	top_back.size = Vector2(1280, 91)
	top_back.color = Color("#050914e8")
	top_back.mouse_filter = Control.MOUSE_FILTER_IGNORE
	hud.add_child(top_back)

	_make_label(hud, "INTEGRITY", Vector2(27, 10), Vector2(200, 18), 10, C_MUTED)
	hp_bar = ProgressBar.new()
	hp_bar.position = Vector2(27, 32)
	hp_bar.size = Vector2(300, 17)
	hp_bar.show_percentage = false
	hp_bar.add_theme_stylebox_override("background", _panel_style(Color("#111827")))
	hp_bar.add_theme_stylebox_override("fill", _panel_style(C_RED))
	hud.add_child(hp_bar)
	hp_label = _make_label(hud, "100 / 100", Vector2(27, 52), Vector2(300, 20), 11, C_TEXT)

	_make_label(hud, "NEURAL SATURATION", Vector2(349, 10), Vector2(260, 18), 10, C_MUTED)
	xp_bar = ProgressBar.new()
	xp_bar.position = Vector2(349, 32)
	xp_bar.size = Vector2(265, 10)
	xp_bar.show_percentage = false
	xp_bar.add_theme_stylebox_override("background", _panel_style(Color("#101827")))
	xp_bar.add_theme_stylebox_override("fill", _panel_style(C_CYAN))
	hud.add_child(xp_bar)
	level_label = _make_label(hud, "LEVEL 01", Vector2(349, 48), Vector2(265, 22), 12, C_CYAN)

	timer_label = _make_label(hud, "00:00", Vector2(540, 9), Vector2(200, 35), 25, C_TEXT, HORIZONTAL_ALIGNMENT_CENTER)
	director_label = _make_label(hud, "THREAT // DORMANT", Vector2(510, 45), Vector2(260, 19), 10, C_MUTED, HORIZONTAL_ALIGNMENT_CENTER)

	score_label = _make_label(hud, "SCORE 000000\nKILLS 000", Vector2(965, 11), Vector2(286, 43), 12, C_TEXT, HORIZONTAL_ALIGNMENT_RIGHT)
	dash_label = _make_label(hud, "DASH // READY", Vector2(965, 57), Vector2(286, 18), 10, C_LIME, HORIZONTAL_ALIGNMENT_RIGHT)

	var bottom_help := _make_label(
		hud,
		"WASD  MOVE     MOUSE  AIM     LMB / J  FIRE     SHIFT / SPACE  DASH     ESC  PAUSE",
		Vector2(24, 681),
		Vector2(1232, 22),
		10,
		C_MUTED,
		HORIZONTAL_ALIGNMENT_CENTER
	)
	bottom_help.add_theme_color_override("font_shadow_color", C_BG)
	bottom_help.add_theme_constant_override("shadow_offset_x", 2)
	bottom_help.add_theme_constant_override("shadow_offset_y", 2)

	boss_label = _make_label(hud, "OVERSEER // UNMATERIALIZED", Vector2(390, 99), Vector2(500, 20), 11, C_MAGENTA, HORIZONTAL_ALIGNMENT_CENTER)
	boss_bar = ProgressBar.new()
	boss_bar.position = Vector2(390, 122)
	boss_bar.size = Vector2(500, 10)
	boss_bar.show_percentage = false
	boss_bar.add_theme_stylebox_override("background", _panel_style(Color("#161224")))
	boss_bar.add_theme_stylebox_override("fill", _panel_style(C_MAGENTA))
	hud.add_child(boss_bar)
	boss_label.visible = false
	boss_bar.visible = false

	announcement_label = _make_label(hud, "", Vector2(220, 155), Vector2(840, 70), 26, C_CYAN, HORIZONTAL_ALIGNMENT_CENTER)
	announcement_label.add_theme_color_override("font_shadow_color", Color(C_CYAN, 0.35))
	announcement_label.add_theme_constant_override("shadow_offset_x", 3)
	announcement_label.add_theme_constant_override("shadow_offset_y", 3)
	announcement_label.visible = false

	_build_title_overlay()
	_build_pause_overlay()
	_build_level_overlay()
	_build_outcome_overlay()
	hud.visible = false


func _build_title_overlay() -> void:
	title_overlay = _make_overlay()
	(title_overlay.get_child(0) as ColorRect).color = Color("#02040a9c")
	var left_rule := ColorRect.new()
	left_rule.position = Vector2(77, 100)
	left_rule.size = Vector2(3, 503)
	left_rule.color = C_CYAN
	title_overlay.add_child(left_rule)
	_make_label(title_overlay, "ABSORBING-STATE // ARENA 01", Vector2(101, 89), Vector2(720, 24), 12, C_CYAN)
	_make_label(title_overlay, "NULLVECTOR", Vector2(98, 121), Vector2(720, 72), 54, C_TEXT)
	_make_label(title_overlay, "NEURAL EXTERMINATION PROTOCOL", Vector2(102, 189), Vector2(660, 30), 19, C_MAGENTA)

	var hash := str(registry.get("model_hash", "NO MODEL"))
	_make_label(
		title_overlay,
		"COMBATANTS SYNTHESIZED BY CATEGORICAL DIFFUSION\nMODEL HASH  " + hash.to_upper() + "\n16 BAKED GENOMES  //  4 HOSTILE LATENT FAMILIES",
		Vector2(103, 239),
		Vector2(560, 83),
		12,
		Color("#a9b8cf")
	)

	var mission := Label.new()
	mission.position = Vector2(103, 349)
	mission.size = Vector2(520, 90)
	mission.text = "SURVIVE THE CONVERGENCE.\nABSORB SHARDS. REWRITE YOUR BUILD.\nDESTROY THE OVERSEER."
	mission.add_theme_font_size_override("font_size", 16)
	mission.add_theme_color_override("font_color", C_TEXT)
	mission.add_theme_constant_override("line_spacing", 6)
	title_overlay.add_child(mission)

	var start_button := Button.new()
	start_button.position = Vector2(102, 477)
	start_button.size = Vector2(355, 58)
	start_button.text = "INITIALIZE RUN   [ ENTER ]"
	_style_button(start_button, C_CYAN)
	start_button.pressed.connect(_start_run)
	title_overlay.add_child(start_button)
	_make_label(title_overlay, "BOSS SIGNAL LOCKS AT 03:00", Vector2(104, 548), Vector2(430, 22), 11, C_MUTED)
	_make_label(title_overlay, "NATIVE GODOT RUNTIME  //  NO WEB LAYER", Vector2(814, 620), Vector2(386, 24), 11, C_LIME, HORIZONTAL_ALIGNMENT_RIGHT)


func _build_pause_overlay() -> void:
	pause_overlay = _make_overlay()
	_make_label(pause_overlay, "SIMULATION SUSPENDED", Vector2(340, 215), Vector2(600, 60), 32, C_TEXT, HORIZONTAL_ALIGNMENT_CENTER)
	_make_label(pause_overlay, "THE LATENT FIELD IS FROZEN", Vector2(340, 273), Vector2(600, 24), 12, C_CYAN, HORIZONTAL_ALIGNMENT_CENTER)
	var resume_button := Button.new()
	resume_button.position = Vector2(470, 337)
	resume_button.size = Vector2(340, 55)
	resume_button.text = "RESUME   [ ESC ]"
	_style_button(resume_button, C_CYAN)
	resume_button.pressed.connect(_resume_run)
	pause_overlay.add_child(resume_button)
	var restart_button := Button.new()
	restart_button.position = Vector2(470, 409)
	restart_button.size = Vector2(340, 48)
	restart_button.text = "RESTART RUN"
	_style_button(restart_button, C_MAGENTA)
	restart_button.pressed.connect(_start_run)
	pause_overlay.add_child(restart_button)
	pause_overlay.visible = false


func _build_level_overlay() -> void:
	level_overlay = _make_overlay()
	level_title = _make_label(level_overlay, "LATENT REWRITE // LEVEL 02", Vector2(220, 104), Vector2(840, 45), 28, C_CYAN, HORIZONTAL_ALIGNMENT_CENTER)
	_make_label(level_overlay, "CHOOSE ONE MUTATION", Vector2(220, 149), Vector2(840, 25), 11, C_MUTED, HORIZONTAL_ALIGNMENT_CENTER)
	for index in range(3):
		var button := Button.new()
		button.position = Vector2(125 + index * 350, 232)
		button.size = Vector2(330, 250)
		button.text_overrun_behavior = TextServer.OVERRUN_TRIM_ELLIPSIS
		button.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		button.alignment = HORIZONTAL_ALIGNMENT_CENTER
		_style_button(button, [C_CYAN, C_MAGENTA, C_LIME][index])
		button.pressed.connect(_select_upgrade.bind(index))
		level_overlay.add_child(button)
		upgrade_buttons.append(button)
	_make_label(level_overlay, "1 / 2 / 3 ALSO SELECT", Vector2(220, 525), Vector2(840, 24), 10, C_MUTED, HORIZONTAL_ALIGNMENT_CENTER)
	level_overlay.visible = false


func _build_outcome_overlay() -> void:
	outcome_overlay = _make_overlay()
	outcome_title = _make_label(outcome_overlay, "SIGNAL LOST", Vector2(260, 180), Vector2(760, 70), 42, C_RED, HORIZONTAL_ALIGNMENT_CENTER)
	outcome_stats = _make_label(outcome_overlay, "", Vector2(340, 260), Vector2(600, 105), 15, C_TEXT, HORIZONTAL_ALIGNMENT_CENTER)
	var restart_button := Button.new()
	restart_button.position = Vector2(470, 417)
	restart_button.size = Vector2(340, 58)
	restart_button.text = "RECOMPILE RUN   [ ENTER ]"
	_style_button(restart_button, C_CYAN)
	restart_button.pressed.connect(_start_run)
	outcome_overlay.add_child(restart_button)
	outcome_overlay.visible = false


func _spawn_title_cast() -> void:
	_clear_actor_nodes()
	camera.position = Vector2.ZERO
	var archetypes := ["dart", "hound", "oracle", "bulwark"]
	var positions := [Vector2(220, -152), Vector2(450, -53), Vector2(218, 85), Vector2(448, 185)]
	for index in range(archetypes.size()):
		var actor := NeuralSpriteClass.new() as NeuralSprite
		actor.configure(_manifest_for(archetypes[index]))
		add_child(actor)
		actor.position = positions[index]
		actor.scale = Vector2.ONE * (2.5 if archetypes[index] != "bulwark" else 2.9)
		actor.set_animation("move", true)
		title_sprites.append(actor)
		title_origins.append(positions[index])


func _clear_actor_nodes() -> void:
	if is_instance_valid(player_sprite):
		player_sprite.queue_free()
	player_sprite = null
	for actor in title_sprites:
		if is_instance_valid(actor):
			actor.queue_free()
	title_sprites.clear()
	title_origins.clear()
	for enemy in enemies:
		if is_instance_valid(enemy.sprite):
			enemy.sprite.queue_free()
	enemies.clear()
	projectiles.clear()
	pickups.clear()
	particles.clear()
	floaters.clear()
	boss_ref = null


func _start_run() -> void:
	_clear_actor_nodes()
	run_seed = rng.randi()
	rng.seed = run_seed
	state = GameState.RUNNING
	player_pos = Vector2.ZERO
	player_velocity = Vector2.ZERO
	aim_direction = Vector2.UP
	hp = 100.0
	max_hp = 100.0
	move_speed = 250.0
	fire_interval = 0.18
	fire_cooldown = 0.05
	bullet_damage = 16.0
	bullet_speed = 820.0
	bullet_radius = 4.0
	bullet_pierce = 0
	projectile_count = 1
	crit_chance = 0.05
	pickup_radius = 115.0
	regeneration = 0.0
	dash_cooldown_max = 1.65
	dash_cooldown = 0.0
	dash_time = 0.0
	dash_serial = 0
	dash_damage = 38.0
	invulnerable = 0.7
	elapsed = 0.0
	spawn_clock = 0.35
	kills = 0
	score = 0
	level = 1
	xp = 0
	xp_needed = 12
	pending_levels = 0
	boss_spawned = false
	announcement_time = 2.5
	announcement_text = "RUN SEED // %08X" % run_seed
	camera_shake = 0.0
	upgrade_levels.clear()

	player_sprite = NeuralSpriteClass.new() as NeuralSprite
	player_sprite.configure(_manifest_for("dart"))
	add_child(player_sprite)
	player_sprite.position = player_pos
	player_sprite.scale = Vector2.ONE * 2.35
	player_sprite.set_animation("idle", true)

	camera.position = player_pos
	title_overlay.visible = false
	pause_overlay.visible = false
	level_overlay.visible = false
	outcome_overlay.visible = false
	hud.visible = true
	boss_label.visible = false
	boss_bar.visible = false
	_update_hud()
	queue_redraw()


func _start_smoke_test() -> void:
	_start_run()
	for archetype in ["dart", "hound", "oracle", "bulwark"]:
		var enemy := _spawn_enemy(archetype)
		enemy.pos = player_pos + Vector2.from_angle(enemies.size() * TAU / 4.0) * 330.0
	if not enemies.is_empty():
		enemies[0].pos = player_pos + Vector2(105.0, 0.0)
		aim_direction = Vector2.RIGHT
		_fire_player_weapon()
	_gain_xp(xp_needed)
	_select_upgrade(0)
	elapsed = BOSS_TIME - 0.25


func _resume_run() -> void:
	if state == GameState.PAUSED:
		state = GameState.RUNNING
		pause_overlay.visible = false


func _unhandled_input(event: InputEvent) -> void:
	if not (event is InputEventKey) or not event.pressed or event.echo:
		return
	var key := event as InputEventKey
	if state == GameState.TITLE and key.keycode == KEY_ENTER:
		_start_run()
	elif state in [GameState.DEAD, GameState.VICTORY] and key.keycode == KEY_ENTER:
		_start_run()
	elif state == GameState.LEVEL_UP and key.keycode in [KEY_1, KEY_2, KEY_3]:
		_select_upgrade(key.keycode - KEY_1)
	elif state == GameState.RUNNING and key.keycode in [KEY_ESCAPE, KEY_P]:
		state = GameState.PAUSED
		pause_overlay.visible = true
	elif state == GameState.PAUSED and key.keycode in [KEY_ESCAPE, KEY_P]:
		_resume_run()


func _process(delta: float) -> void:
	if state == GameState.TITLE:
		_update_title_cast(delta)
		queue_redraw()
		return
	if state != GameState.RUNNING:
		queue_redraw()
		return
	_update_game(delta)


func _update_title_cast(delta: float) -> void:
	var now := Time.get_ticks_msec() * 0.001
	for index in range(title_sprites.size()):
		var actor: Node2D = title_sprites[index]
		if not is_instance_valid(actor):
			continue
		actor.position = title_origins[index] + Vector2(sin(now * 0.7 + index) * 10.0, cos(now * 1.1 + index * 0.8) * 14.0)
		actor.rotation += delta * (0.12 if index % 2 == 0 else -0.1)


func _update_game(delta: float) -> void:
	elapsed += delta
	fire_cooldown -= delta
	dash_cooldown = maxf(0.0, dash_cooldown - delta)
	dash_time = maxf(0.0, dash_time - delta)
	invulnerable = maxf(0.0, invulnerable - delta)
	announcement_time = maxf(0.0, announcement_time - delta)
	hp = minf(max_hp, hp + regeneration * delta)

	_update_player(delta)
	_update_spawn_director(delta)
	_update_enemies(delta)
	_update_projectiles(delta)
	_update_pickups(delta)
	_update_particles(delta)
	_remove_dead_enemies()

	if is_instance_valid(player_sprite):
		player_sprite.position = player_pos
	camera.position = player_pos
	if camera_shake > 0.01:
		camera.offset = Vector2.from_angle(rng.randf_range(0.0, TAU)) * camera_shake
		camera_shake = maxf(0.0, camera_shake - delta * 34.0)
	else:
		camera.offset = Vector2.ZERO

	_update_hud()
	queue_redraw()


func _update_player(delta: float) -> void:
	var move_input := Input.get_vector("move_left", "move_right", "move_up", "move_down")
	var mouse_delta := get_global_mouse_position() - player_pos
	if mouse_delta.length_squared() > 9.0:
		aim_direction = mouse_delta.normalized()

	if Input.is_action_just_pressed("dash") and dash_cooldown <= 0.0:
		var dash_direction := move_input.normalized() if move_input.length_squared() > 0.01 else aim_direction
		player_velocity = dash_direction * 950.0
		dash_time = 0.18
		dash_cooldown = dash_cooldown_max
		invulnerable = 0.32
		dash_serial += 1
		camera_shake = 7.0
		_spawn_burst(player_pos, C_CYAN, 18, 260.0, 0.42, true)

	if dash_time > 0.0:
		player_pos += player_velocity * delta
		if rng.randf() < 0.9:
			_spawn_particle(player_pos - player_velocity.normalized() * 17.0, -player_velocity * 0.08, C_CYAN, 0.22, 4.0, true)
		for enemy in enemies:
			if enemy.dead or enemy.dash_stamp == dash_serial:
				continue
			if player_pos.distance_squared_to(enemy.pos) <= pow(PLAYER_RADIUS + enemy.radius + 8.0, 2):
				enemy.dash_stamp = dash_serial
				_damage_enemy(enemy, dash_damage, player_velocity.normalized() * 250.0, true)
	else:
		var target_velocity := move_input * move_speed
		player_velocity = player_velocity.lerp(target_velocity, 1.0 - exp(-delta * 17.0))
		player_pos += player_velocity * delta

	player_pos.x = clampf(player_pos.x, ARENA_RECT.position.x + 32.0, ARENA_RECT.end.x - 32.0)
	player_pos.y = clampf(player_pos.y, ARENA_RECT.position.y + 32.0, ARENA_RECT.end.y - 32.0)

	if is_instance_valid(player_sprite):
		player_sprite.rotation = aim_direction.angle() + PI * 0.5
		if dash_time > 0.0 or move_input.length_squared() > 0.02:
			player_sprite.set_animation("move")
		elif player_sprite.animation_name == "move":
			player_sprite.set_animation("idle")
		player_sprite.modulate = Color(1.0, 1.0, 1.0, 0.45 if invulnerable > 0.0 and int(invulnerable * 30.0) % 2 == 0 else 1.0)

	if Input.is_action_pressed("attack") and fire_cooldown <= 0.0:
		_fire_player_weapon()


func _fire_player_weapon() -> void:
	fire_cooldown = fire_interval
	if is_instance_valid(player_sprite):
		player_sprite.set_animation("attack", true)
	var spread_step := deg_to_rad(8.0)
	for index in range(projectile_count):
		var centered_index := float(index) - float(projectile_count - 1) * 0.5
		var direction := aim_direction.rotated(centered_index * spread_step)
		var bullet := ProjectileData.new()
		bullet.pos = player_pos + direction * 31.0
		bullet.previous = bullet.pos
		bullet.velocity = direction * bullet_speed
		bullet.damage = bullet_damage
		if rng.randf() < crit_chance:
			bullet.damage *= 2.0
			bullet.color = C_LIME
		else:
			bullet.color = C_CYAN
		bullet.radius = bullet_radius
		bullet.life = 1.45
		bullet.friendly = true
		bullet.pierce = bullet_pierce
		projectiles.append(bullet)
	_spawn_particle(player_pos + aim_direction * 31.0, -aim_direction * 50.0, Color.WHITE, 0.12, 7.0, false)


func _update_spawn_director(delta: float) -> void:
	if not boss_spawned and elapsed >= BOSS_TIME:
		boss_spawned = true
		_spawn_enemy("oracle", true)
		announcement_text = "WARNING // OVERSEER MATERIALIZED"
		announcement_time = 4.0
		camera_shake = 15.0
		return

	spawn_clock -= delta
	if spawn_clock > 0.0 or enemies.size() >= MAX_ENEMIES:
		return
	var interval := maxf(0.19, 0.82 - elapsed * 0.0025)
	if boss_spawned:
		interval *= 1.75
	spawn_clock = interval * rng.randf_range(0.72, 1.22)
	var count := 1
	if elapsed > 90.0 and rng.randf() < 0.28:
		count += 1
	if elapsed > 150.0 and rng.randf() < 0.18:
		count += 1
	for _index in range(count):
		var roll := rng.randf()
		var archetype := "dart"
		if elapsed > 22.0 and roll < 0.34:
			archetype = "hound"
		if elapsed > 52.0 and roll < 0.19:
			archetype = "oracle"
		if elapsed > 86.0 and roll < 0.10:
			archetype = "bulwark"
		_spawn_enemy(archetype)


func _spawn_enemy(archetype: String, as_boss := false) -> EnemyData:
	var enemy := EnemyData.new()
	enemy.archetype = archetype
	enemy.is_boss = as_boss
	var angle := rng.randf_range(0.0, TAU)
	var spawn_distance := rng.randf_range(610.0, 820.0)
	enemy.pos = player_pos + Vector2.from_angle(angle) * spawn_distance
	enemy.pos.x = clampf(enemy.pos.x, ARENA_RECT.position.x + 45.0, ARENA_RECT.end.x - 45.0)
	enemy.pos.y = clampf(enemy.pos.y, ARENA_RECT.position.y + 45.0, ARENA_RECT.end.y - 45.0)
	enemy.phase = rng.randf_range(0.0, 8.0)
	var difficulty := 1.0 + elapsed / 165.0

	match archetype:
		"dart":
			enemy.max_hp = 18.0 * difficulty
			enemy.speed = 155.0 + minf(elapsed * 0.11, 24.0)
			enemy.radius = 19.0
			enemy.damage = 8.0
			enemy.xp_value = 2
			enemy.attack_cooldown = rng.randf_range(0.4, 1.0)
		"hound":
			enemy.max_hp = 38.0 * difficulty
			enemy.speed = 112.0
			enemy.radius = 24.0
			enemy.damage = 12.0
			enemy.xp_value = 4
			enemy.attack_cooldown = rng.randf_range(1.5, 2.4)
		"oracle":
			enemy.max_hp = 56.0 * difficulty
			enemy.speed = 82.0
			enemy.radius = 25.0
			enemy.damage = 9.0
			enemy.xp_value = 6
			enemy.attack_cooldown = rng.randf_range(0.8, 1.7)
		"bulwark":
			enemy.max_hp = 125.0 * difficulty
			enemy.speed = 48.0
			enemy.radius = 33.0
			enemy.damage = 18.0
			enemy.xp_value = 11
			enemy.attack_cooldown = rng.randf_range(1.4, 2.6)

	if as_boss:
		enemy.archetype = "overseer"
		enemy.max_hp = 2200.0 + level * 115.0
		enemy.speed = 64.0
		enemy.radius = 62.0
		enemy.damage = 22.0
		enemy.xp_value = 100
		enemy.attack_cooldown = 1.1

	enemy.hp = enemy.max_hp
	enemy.sprite = NeuralSpriteClass.new() as NeuralSprite
	enemy.sprite.configure(_manifest_for(archetype))
	add_child(enemy.sprite)
	enemy.sprite.position = enemy.pos
	enemy.sprite.scale = Vector2.ONE * (4.2 if as_boss else (2.35 if archetype == "bulwark" else 1.9))
	enemy.sprite.set_animation("move", true)
	enemies.append(enemy)
	if as_boss:
		boss_ref = enemy
		boss_label.visible = true
		boss_bar.visible = true
	_spawn_burst(enemy.pos, C_MAGENTA if as_boss else C_BLUE, 28 if as_boss else 8, 260.0, 0.72, true)
	return enemy


func _update_enemies(delta: float) -> void:
	for enemy in enemies:
		if enemy.dead:
			continue
		enemy.attack_cooldown -= delta
		enemy.contact_cooldown = maxf(0.0, enemy.contact_cooldown - delta)
		enemy.phase += delta
		enemy.flash = maxf(0.0, enemy.flash - delta * 7.0)
		var to_player: Vector2 = player_pos - enemy.pos
		var distance := maxf(1.0, to_player.length())
		var direction: Vector2 = to_player / distance

		if enemy.is_boss:
			_update_boss(enemy, direction, distance, delta)
		else:
			match enemy.archetype:
				"dart":
					var weave: Vector2 = direction.rotated(PI * 0.5) * sin(enemy.phase * 6.5) * 0.42
					enemy.velocity = enemy.velocity.lerp((direction + weave).normalized() * enemy.speed, 1.0 - exp(-delta * 8.0))
				"hound":
					_update_hound(enemy, direction, delta)
				"oracle":
					_update_oracle(enemy, direction, distance, delta)
				"bulwark":
					_update_bulwark(enemy, direction, distance, delta)

		enemy.pos += enemy.velocity * delta
		enemy.pos.x = clampf(enemy.pos.x, ARENA_RECT.position.x + enemy.radius, ARENA_RECT.end.x - enemy.radius)
		enemy.pos.y = clampf(enemy.pos.y, ARENA_RECT.position.y + enemy.radius, ARENA_RECT.end.y - enemy.radius)
		if is_instance_valid(enemy.sprite):
			enemy.sprite.position = enemy.pos
			if enemy.velocity.length_squared() > 4.0:
				enemy.sprite.rotation = enemy.velocity.angle() + PI * 0.5
			if enemy.flash > 0.0:
				enemy.sprite.modulate = Color(1.8, 1.8, 1.8, 1.0)
			else:
				enemy.sprite.modulate = Color.WHITE

		if distance <= PLAYER_RADIUS + enemy.radius and enemy.contact_cooldown <= 0.0:
			enemy.contact_cooldown = 0.75
			_damage_player(enemy.damage, direction)

	_resolve_enemy_separation()


func _update_hound(enemy: EnemyData, direction: Vector2, delta: float) -> void:
	if enemy.mode == 0:
		enemy.velocity = enemy.velocity.lerp(direction * enemy.speed, 1.0 - exp(-delta * 7.0))
		if enemy.attack_cooldown <= 0.0:
			enemy.mode = 1
			enemy.attack_cooldown = 0.42
			enemy.charge_direction = direction
			enemy.velocity *= 0.1
			enemy.sprite.set_animation("attack", true)
	elif enemy.mode == 1:
		enemy.velocity = enemy.velocity.lerp(Vector2.ZERO, 1.0 - exp(-delta * 12.0))
		if enemy.attack_cooldown <= 0.0:
			enemy.mode = 2
			enemy.attack_cooldown = 0.64
			enemy.velocity = enemy.charge_direction * 430.0
			_spawn_burst(enemy.pos, C_RED, 7, 120.0, 0.3, true)
	else:
		enemy.velocity = enemy.velocity.lerp(enemy.charge_direction * 360.0, 1.0 - exp(-delta * 3.0))
		if enemy.attack_cooldown <= 0.0:
			enemy.mode = 0
			enemy.attack_cooldown = rng.randf_range(1.7, 2.6)
			enemy.sprite.set_animation("move")


func _update_oracle(enemy: EnemyData, direction: Vector2, distance: float, delta: float) -> void:
	var radial := 0.0
	if distance > 390.0:
		radial = 1.0
	elif distance < 265.0:
		radial = -1.0
	var orbit := direction.rotated(PI * 0.5) * (1.0 if int(enemy.phase * 0.35) % 2 == 0 else -1.0)
	var desired := (direction * radial + orbit * 0.75).normalized() * enemy.speed
	enemy.velocity = enemy.velocity.lerp(desired, 1.0 - exp(-delta * 4.5))
	if enemy.attack_cooldown <= 0.0 and distance < 650.0:
		enemy.attack_cooldown = rng.randf_range(1.35, 1.9)
		enemy.sprite.set_animation("attack", true)
		_spawn_enemy_bullet(enemy.pos + direction * 24.0, direction, enemy.damage, 280.0, C_MAGENTA, 6.0)


func _update_bulwark(enemy: EnemyData, direction: Vector2, distance: float, delta: float) -> void:
	enemy.velocity = enemy.velocity.lerp(direction * enemy.speed, 1.0 - exp(-delta * 3.0))
	if enemy.attack_cooldown <= 0.0 and distance < 570.0:
		enemy.attack_cooldown = rng.randf_range(2.15, 2.85)
		enemy.sprite.set_animation("attack", true)
		for offset in [-0.22, 0.0, 0.22]:
			_spawn_enemy_bullet(enemy.pos + direction * 34.0, direction.rotated(offset), enemy.damage * 0.72, 235.0, C_RED, 7.0)


func _update_boss(enemy: EnemyData, direction: Vector2, distance: float, delta: float) -> void:
	var orbit := direction.rotated(PI * 0.5) * sin(enemy.phase * 0.55)
	var radial := 1.0 if distance > 410.0 else -0.65
	enemy.velocity = enemy.velocity.lerp((direction * radial + orbit).normalized() * enemy.speed, 1.0 - exp(-delta * 2.8))
	if enemy.attack_cooldown <= 0.0:
		enemy.sprite.set_animation("attack", true)
		var pattern := int(enemy.phase) % 3
		if pattern == 0:
			for index in range(12):
				_spawn_enemy_bullet(enemy.pos, Vector2.from_angle(index * TAU / 12.0 + enemy.phase * 0.1), 11.0, 215.0, C_MAGENTA, 7.0)
			enemy.attack_cooldown = 1.45
		elif pattern == 1:
			for offset in [-0.24, -0.12, 0.0, 0.12, 0.24]:
				_spawn_enemy_bullet(enemy.pos + direction * 52.0, direction.rotated(offset), 14.0, 330.0, C_RED, 7.0)
			enemy.attack_cooldown = 1.05
		else:
			for offset in [-0.08, 0.08]:
				_spawn_enemy_bullet(enemy.pos + direction * 52.0, direction.rotated(offset), 17.0, 410.0, C_LIME, 8.0)
			enemy.attack_cooldown = 0.82

	if int(enemy.phase) % 8 == 0 and enemy.mode != int(enemy.phase):
		enemy.mode = int(enemy.phase)
		if enemies.size() < MAX_ENEMIES - 4:
			for index in range(3):
				var add_type: String = ["dart", "hound", "oracle"][index]
				var add := _spawn_enemy(add_type)
				add.pos = enemy.pos + Vector2.from_angle(index * TAU / 3.0) * 115.0


func _resolve_enemy_separation() -> void:
	for index in range(enemies.size()):
		var first: EnemyData = enemies[index]
		if first.dead:
			continue
		for other_index in range(index + 1, enemies.size()):
			var second: EnemyData = enemies[other_index]
			if second.dead:
				continue
			var offset := second.pos - first.pos
			var minimum := (first.radius + second.radius) * 0.72
			var distance_squared := offset.length_squared()
			if distance_squared < minimum * minimum and distance_squared > 0.001:
				var distance := sqrt(distance_squared)
				var push := offset / distance * (minimum - distance) * 0.28
				first.pos -= push
				second.pos += push


func _spawn_enemy_bullet(origin: Vector2, direction: Vector2, damage: float, speed: float, color: Color, radius: float) -> void:
	var bullet := ProjectileData.new()
	bullet.pos = origin
	bullet.previous = origin
	bullet.velocity = direction.normalized() * speed
	bullet.damage = damage
	bullet.radius = radius
	bullet.life = 4.0
	bullet.friendly = false
	bullet.color = color
	projectiles.append(bullet)


func _update_projectiles(delta: float) -> void:
	for index in range(projectiles.size() - 1, -1, -1):
		var bullet: ProjectileData = projectiles[index]
		bullet.life -= delta
		bullet.previous = bullet.pos
		bullet.pos += bullet.velocity * delta
		if bullet.life <= 0.0 or not ARENA_RECT.grow(100.0).has_point(bullet.pos):
			projectiles.remove_at(index)
			continue

		if bullet.friendly:
			var consumed := false
			for enemy in enemies:
				if enemy.dead or bullet.hit_ids.has(enemy.get_instance_id()):
					continue
				if _segment_circle_hit(bullet.previous, bullet.pos, enemy.pos, enemy.radius + bullet.radius):
					bullet.hit_ids[enemy.get_instance_id()] = true
					_damage_enemy(enemy, bullet.damage, bullet.velocity.normalized() * 95.0)
					if bullet.pierce > 0:
						bullet.pierce -= 1
						bullet.damage *= 0.87
					else:
						consumed = true
						break
			if consumed:
				projectiles.remove_at(index)
		else:
			if invulnerable <= 0.0 and _segment_circle_hit(bullet.previous, bullet.pos, player_pos, PLAYER_RADIUS + bullet.radius):
				_damage_player(bullet.damage, bullet.velocity.normalized())
				projectiles.remove_at(index)


func _segment_circle_hit(start: Vector2, finish: Vector2, center: Vector2, radius: float) -> bool:
	var segment := finish - start
	var length_squared := segment.length_squared()
	if length_squared <= 0.0001:
		return start.distance_squared_to(center) <= radius * radius
	var t := clampf((center - start).dot(segment) / length_squared, 0.0, 1.0)
	var closest := start + segment * t
	return closest.distance_squared_to(center) <= radius * radius


func _damage_enemy(enemy: EnemyData, amount: float, knockback: Vector2, dash_hit := false) -> void:
	if enemy.dead:
		return
	enemy.hp -= amount
	enemy.velocity += knockback / maxf(1.0, enemy.radius * 0.04)
	enemy.flash = 1.0
	var damage_text := FloaterData.new()
	damage_text.pos = enemy.pos + Vector2(rng.randf_range(-10.0, 10.0), -enemy.radius)
	damage_text.text = "%d" % int(round(amount))
	damage_text.color = C_LIME if amount > bullet_damage * 1.5 or dash_hit else C_TEXT
	floaters.append(damage_text)
	_spawn_burst(enemy.pos, C_LIME if dash_hit else C_CYAN, 6 if enemy.is_boss else 3, 145.0, 0.28, true)
	if enemy.hp <= 0.0:
		_kill_enemy(enemy)


func _kill_enemy(enemy: EnemyData) -> void:
	if enemy.dead:
		return
	enemy.dead = true
	kills += 1
	var base_score := 500 if enemy.is_boss else enemy.xp_value * 18
	score += base_score
	_spawn_burst(enemy.pos, C_MAGENTA if enemy.is_boss else C_CYAN, 70 if enemy.is_boss else 16, 380.0 if enemy.is_boss else 225.0, 1.0 if enemy.is_boss else 0.55, true)
	if is_instance_valid(enemy.sprite):
		enemy.sprite.set_animation("hit", true)
		enemy.sprite.queue_free()

	if enemy.is_boss:
		boss_ref = null
		camera_shake = 24.0
		_finish_run(true)
		return

	var shards := 1
	if enemy.xp_value >= 6:
		shards = 2
	if enemy.xp_value >= 10:
		shards = 3
	var remaining_value := enemy.xp_value
	for index in range(shards):
		var pickup := PickupData.new()
		pickup.pos = enemy.pos + Vector2.from_angle(index * TAU / shards + rng.randf_range(-0.35, 0.35)) * rng.randf_range(4.0, 18.0)
		pickup.velocity = Vector2.from_angle(rng.randf_range(0.0, TAU)) * rng.randf_range(50.0, 130.0)
		pickup.value = maxi(1, remaining_value / (shards - index))
		remaining_value -= pickup.value
		pickup.phase = rng.randf_range(0.0, TAU)
		pickups.append(pickup)


func _remove_dead_enemies() -> void:
	for index in range(enemies.size() - 1, -1, -1):
		if enemies[index].dead:
			enemies.remove_at(index)


func _damage_player(amount: float, source_direction: Vector2) -> void:
	if invulnerable > 0.0 or state != GameState.RUNNING:
		return
	hp -= amount
	invulnerable = 0.48
	player_velocity += source_direction * 180.0
	camera_shake = minf(15.0, 5.0 + amount * 0.45)
	_spawn_burst(player_pos, C_RED, 14, 230.0, 0.52, true)
	var hit_text := FloaterData.new()
	hit_text.pos = player_pos + Vector2(0.0, -36.0)
	hit_text.text = "-%d" % int(round(amount))
	hit_text.color = C_RED
	floaters.append(hit_text)
	if is_instance_valid(player_sprite):
		player_sprite.set_animation("hit", true)
	if hp <= 0.0:
		hp = 0.0
		_finish_run(false)


func _update_pickups(delta: float) -> void:
	for index in range(pickups.size() - 1, -1, -1):
		var pickup: PickupData = pickups[index]
		pickup.life -= delta
		pickup.phase += delta * 4.0
		pickup.velocity = pickup.velocity.lerp(Vector2.ZERO, 1.0 - exp(-delta * 4.0))
		var to_player := player_pos - pickup.pos
		var distance := to_player.length()
		if distance < pickup_radius and distance > 0.1:
			var pull := lerpf(220.0, 780.0, 1.0 - distance / pickup_radius)
			pickup.velocity = pickup.velocity.lerp(to_player / distance * pull, 1.0 - exp(-delta * 10.0))
		pickup.pos += pickup.velocity * delta
		if distance <= PLAYER_RADIUS + 11.0:
			_gain_xp(pickup.value)
			_spawn_burst(pickup.pos, C_LIME, 5, 100.0, 0.25, false)
			pickups.remove_at(index)
		elif pickup.life <= 0.0:
			pickups.remove_at(index)


func _gain_xp(amount: int) -> void:
	xp += amount
	score += amount * 4
	while xp >= xp_needed:
		xp -= xp_needed
		level += 1
		xp_needed = 10 + level * 7 + int(pow(level, 1.35))
		pending_levels += 1
	if pending_levels > 0 and state == GameState.RUNNING:
		_show_level_up()


func _show_level_up() -> void:
	state = GameState.LEVEL_UP
	level_title.text = "LATENT REWRITE // LEVEL %02d" % level
	offered_upgrades.clear()
	var candidates: Array = []
	for item in upgrade_catalog:
		var item_id := str(item["id"])
		if int(upgrade_levels.get(item_id, 0)) < int(item["max"]):
			candidates.append(item)
	if candidates.is_empty():
		candidates = upgrade_catalog.duplicate()
	while candidates.size() < 3:
		candidates.append(upgrade_catalog[rng.randi_range(0, upgrade_catalog.size() - 1)])
	for index in range(3):
		var pick_index := rng.randi_range(0, candidates.size() - 1)
		var choice: Dictionary = candidates[pick_index]
		candidates.remove_at(pick_index)
		offered_upgrades.append(str(choice["id"]))
		var next_rank := int(upgrade_levels.get(choice["id"], 0)) + 1
		upgrade_buttons[index].text = (
			"0%d\n\n%s\n\n%s\n\nRANK %d / %d"
			% [index + 1, str(choice["name"]), str(choice["description"]), next_rank, int(choice["max"])]
		)
	level_overlay.visible = true
	upgrade_buttons[0].grab_focus()


func _select_upgrade(index: int) -> void:
	if state != GameState.LEVEL_UP or index < 0 or index >= offered_upgrades.size():
		return
	var upgrade_id := offered_upgrades[index]
	upgrade_levels[upgrade_id] = int(upgrade_levels.get(upgrade_id, 0)) + 1
	match upgrade_id:
		"overclock":
			fire_interval *= 0.82
		"rail_coils":
			bullet_damage *= 1.28
		"forked_signal":
			projectile_count += 1
		"phase_bore":
			bullet_pierce += 1
		"flux_drive":
			move_speed *= 1.14
		"capacitor":
			max_hp += 25.0
			hp = minf(max_hp, hp + 25.0)
		"magnetism":
			pickup_radius += 65.0
		"phase_lens":
			bullet_speed *= 1.16
			bullet_radius += 1.0
		"nanoshield":
			regeneration += 0.7
		"criticality":
			crit_chance += 0.09
		"nova_dash":
			dash_damage *= 1.45
			dash_cooldown_max *= 0.88
	pending_levels -= 1
	level_overlay.visible = false
	announcement_text = "GENOME PATCHED // " + upgrade_id.replace("_", " ").to_upper()
	announcement_time = 1.8
	if pending_levels > 0:
		_show_level_up()
	else:
		state = GameState.RUNNING
	_update_hud()


func _finish_run(victory: bool) -> void:
	state = GameState.VICTORY if victory else GameState.DEAD
	level_overlay.visible = false
	pause_overlay.visible = false
	outcome_overlay.visible = true
	outcome_title.text = "OVERSEER ERASED" if victory else "SIGNAL LOST"
	outcome_title.add_theme_color_override("font_color", C_LIME if victory else C_RED)
	outcome_stats.text = (
		"%s\n\nSURVIVAL  %s     LEVEL  %02d\nKILLS  %03d     SCORE  %06d"
		% ["LATENT FIELD STABILIZED" if victory else "THE SWARM CONSUMED THIS GENOME", _format_time(elapsed), level, kills, score]
	)
	boss_bar.visible = false
	boss_label.visible = false


func _format_time(value: float) -> String:
	var whole := int(value)
	return "%02d:%02d" % [whole / 60, whole % 60]


func _update_hud() -> void:
	if not is_instance_valid(hp_bar):
		return
	hp_bar.max_value = max_hp
	hp_bar.value = hp
	hp_label.text = "%03d / %03d" % [int(ceil(hp)), int(max_hp)]
	xp_bar.max_value = xp_needed
	xp_bar.value = xp
	level_label.text = "LEVEL %02d     %d / %d XP" % [level, xp, xp_needed]
	timer_label.text = _format_time(elapsed)
	score_label.text = "SCORE %06d\nKILLS %03d" % [score, kills]
	var threat := "DORMANT"
	if elapsed > 20.0:
		threat = "RISING"
	if elapsed > 65.0:
		threat = "SEVERE"
	if elapsed > 120.0:
		threat = "CRITICAL"
	if boss_spawned:
		threat = "OVERSEER"
	director_label.text = "THREAT // %s     HOSTILES %02d" % [threat, enemies.size()]
	if dash_cooldown <= 0.0:
		dash_label.text = "DASH // READY"
		dash_label.add_theme_color_override("font_color", C_LIME)
	else:
		dash_label.text = "DASH // %.1fs" % dash_cooldown
		dash_label.add_theme_color_override("font_color", C_MUTED)
	if boss_ref != null and not boss_ref.dead:
		boss_bar.max_value = boss_ref.max_hp
		boss_bar.value = boss_ref.hp
		boss_label.text = "OVERSEER // %d%%" % int(ceil(boss_ref.hp / boss_ref.max_hp * 100.0))
	announcement_label.visible = announcement_time > 0.0
	announcement_label.text = announcement_text
	if announcement_time > 0.0:
		announcement_label.modulate.a = minf(1.0, announcement_time * 1.5)


func _spawn_particle(origin: Vector2, velocity: Vector2, color: Color, life: float, size: float, streak: bool) -> void:
	var particle := ParticleData.new()
	particle.pos = origin
	particle.velocity = velocity
	particle.life = life
	particle.max_life = life
	particle.color = color
	particle.size = size
	particle.streak = streak
	particles.append(particle)


func _spawn_burst(origin: Vector2, color: Color, count: int, speed: float, life: float, streak: bool) -> void:
	for _index in range(count):
		var direction := Vector2.from_angle(rng.randf_range(0.0, TAU))
		_spawn_particle(origin, direction * rng.randf_range(speed * 0.25, speed), color.lerp(Color.WHITE, rng.randf_range(0.0, 0.32)), rng.randf_range(life * 0.5, life), rng.randf_range(1.5, 5.5), streak)


func _update_particles(delta: float) -> void:
	for index in range(particles.size() - 1, -1, -1):
		var particle: ParticleData = particles[index]
		particle.life -= delta
		particle.pos += particle.velocity * delta
		particle.velocity *= exp(-delta * 3.4)
		if particle.life <= 0.0:
			particles.remove_at(index)
	for index in range(floaters.size() - 1, -1, -1):
		var floater: FloaterData = floaters[index]
		floater.life -= delta
		floater.pos.y -= delta * 38.0
		if floater.life <= 0.0:
			floaters.remove_at(index)


func _draw() -> void:
	draw_rect(ARENA_RECT, C_BG, true)
	_draw_grid()
	if state == GameState.TITLE:
		_draw_title_field()
		return

	_draw_arena_markings()
	_draw_pickups()
	_draw_projectiles()
	_draw_particles()
	_draw_enemy_status()
	_draw_player_fx()
	_draw_floaters()
	_draw_crosshair()


func _draw_grid() -> void:
	for x in range(int(ARENA_RECT.position.x), int(ARENA_RECT.end.x) + 1, 32):
		var strong := posmod(x, 128) == 0
		var color := Color(0.18, 0.52, 0.72, 0.072 if strong else 0.025)
		draw_line(Vector2(x, ARENA_RECT.position.y), Vector2(x, ARENA_RECT.end.y), color, 1.0)
	for y in range(int(ARENA_RECT.position.y), int(ARENA_RECT.end.y) + 1, 32):
		var strong := posmod(y, 128) == 0
		var color := Color(0.18, 0.52, 0.72, 0.072 if strong else 0.025)
		draw_line(Vector2(ARENA_RECT.position.x, y), Vector2(ARENA_RECT.end.x, y), color, 1.0)


func _draw_title_field() -> void:
	var now := Time.get_ticks_msec() * 0.001
	for ring in range(5):
		var radius := 205.0 + ring * 72.0 + sin(now * 0.7 + ring) * 7.0
		draw_arc(Vector2(335, 20), radius, 0.0, TAU, 96, Color(C_CYAN, 0.035 + ring * 0.006), 1.0)
	for index in range(title_origins.size()):
		var origin := title_origins[index]
		var accent: Color = [C_CYAN, C_RED, C_MAGENTA, C_LIME][index]
		draw_circle(origin, 50.0 + sin(now * 2.0 + index) * 5.0, Color(accent, 0.055), false, 2.0)
		draw_line(origin + Vector2(-52, 53), origin + Vector2(52, 53), Color(accent, 0.5), 1.0)
		draw_string(ThemeDB.fallback_font, origin + Vector2(-51, 71), ["DART // RUSH", "HOUND // CHARGE", "ORACLE // RANGE", "BULWARK // SIEGE"][index], HORIZONTAL_ALIGNMENT_LEFT, -1, 10, Color(accent, 0.88))


func _draw_arena_markings() -> void:
	draw_rect(ARENA_RECT, Color(C_CYAN, 0.42), false, 3.0)
	draw_rect(ARENA_RECT.grow(-10.0), Color(C_BLUE, 0.15), false, 1.0)
	var corner_length := 80.0
	for corner in [ARENA_RECT.position, Vector2(ARENA_RECT.end.x, ARENA_RECT.position.y), ARENA_RECT.end, Vector2(ARENA_RECT.position.x, ARENA_RECT.end.y)]:
		var inward_x := 1.0 if corner.x == ARENA_RECT.position.x else -1.0
		var inward_y := 1.0 if corner.y == ARENA_RECT.position.y else -1.0
		draw_line(corner, corner + Vector2(inward_x * corner_length, 0), C_CYAN, 4.0)
		draw_line(corner, corner + Vector2(0, inward_y * corner_length), C_CYAN, 4.0)
	for marker in [Vector2(-800, -500), Vector2(800, -500), Vector2(-800, 500), Vector2(800, 500)]:
		draw_circle(marker, 42.0, Color(C_MAGENTA, 0.04), false, 1.0)
		draw_line(marker + Vector2(-11, 0), marker + Vector2(11, 0), Color(C_MAGENTA, 0.26), 1.0)
		draw_line(marker + Vector2(0, -11), marker + Vector2(0, 11), Color(C_MAGENTA, 0.26), 1.0)


func _draw_pickups() -> void:
	for pickup in pickups:
		var pulse := 0.72 + sin(pickup.phase) * 0.22
		var points := PackedVector2Array([
			pickup.pos + Vector2(0, -8),
			pickup.pos + Vector2(7, 0),
			pickup.pos + Vector2(0, 8),
			pickup.pos + Vector2(-7, 0),
		])
		draw_circle(pickup.pos, 17.0 + pulse * 4.0, Color(C_LIME, 0.06), true)
		draw_colored_polygon(points, Color(C_LIME, pulse))
		draw_polyline(PackedVector2Array([points[0], points[1], points[2], points[3], points[0]]), Color.WHITE, 1.0)


func _draw_projectiles() -> void:
	for bullet in projectiles:
		var direction: Vector2 = bullet.velocity.normalized()
		var tail: Vector2 = bullet.pos - direction * (22.0 if bullet.friendly else 12.0)
		draw_line(tail, bullet.pos, Color(bullet.color, 0.12), bullet.radius * 4.2)
		draw_line(tail, bullet.pos, Color(bullet.color, 0.55), bullet.radius * 1.8)
		draw_line(tail, bullet.pos, Color.WHITE, maxf(1.0, bullet.radius * 0.48))
		draw_circle(bullet.pos, bullet.radius * 0.7, Color.WHITE, true)


func _draw_particles() -> void:
	for particle in particles:
		var alpha := clampf(particle.life / particle.max_life, 0.0, 1.0)
		var color := Color(particle.color, particle.color.a * alpha)
		if particle.streak and particle.velocity.length_squared() > 30.0:
			draw_line(particle.pos, particle.pos - particle.velocity.normalized() * particle.size * 3.0, Color(color, alpha * 0.35), particle.size * 2.2)
			draw_line(particle.pos, particle.pos - particle.velocity.normalized() * particle.size * 2.0, color, maxf(1.0, particle.size * 0.65))
		else:
			draw_circle(particle.pos, particle.size * alpha, color, true)


func _draw_enemy_status() -> void:
	for enemy in enemies:
		if enemy.dead:
			continue
		var ratio := clampf(enemy.hp / enemy.max_hp, 0.0, 1.0)
		if ratio < 0.999 and not enemy.is_boss:
			var width: float = enemy.radius * 1.7
			var y: float = enemy.pos.y - enemy.radius - 14.0
			draw_rect(Rect2(enemy.pos.x - width * 0.5, y, width, 4), Color("#111827"), true)
			draw_rect(Rect2(enemy.pos.x - width * 0.5, y, width * ratio, 4), C_RED, true)
		if enemy.archetype == "hound" and enemy.mode == 1:
			var telegraph_alpha := 0.22 + 0.22 * sin(enemy.phase * 24.0)
			draw_line(enemy.pos, enemy.pos + enemy.charge_direction * 245.0, Color(C_RED, telegraph_alpha), 3.0)
		if enemy.is_boss:
			draw_circle(enemy.pos, enemy.radius + 25.0 + sin(enemy.phase * 3.0) * 7.0, Color(C_MAGENTA, 0.22), false, 2.0)


func _draw_player_fx() -> void:
	var pulse := 0.5 + 0.5 * sin(Time.get_ticks_msec() * 0.006)
	draw_circle(player_pos, PLAYER_RADIUS + 16.0 + pulse * 3.0, Color(C_CYAN, 0.045), true)
	draw_circle(player_pos, PLAYER_RADIUS + 7.0, Color(C_CYAN, 0.22 if invulnerable <= 0.0 else 0.7), false, 2.0)
	if dash_cooldown <= 0.0:
		draw_arc(player_pos, PLAYER_RADIUS + 12.0, -PI * 0.75, PI * 0.75, 28, Color(C_LIME, 0.55), 2.0)
	elif dash_time > 0.0:
		draw_circle(player_pos, PLAYER_RADIUS + 42.0, Color(C_CYAN, 0.2), false, 4.0)


func _draw_floaters() -> void:
	for floater in floaters:
		var alpha := clampf(floater.life / floater.max_life, 0.0, 1.0)
		draw_string(ThemeDB.fallback_font, floater.pos, floater.text, HORIZONTAL_ALIGNMENT_CENTER, 38.0, 15, Color(floater.color, alpha))


func _draw_crosshair() -> void:
	var mouse := get_global_mouse_position()
	var color := C_LIME if Input.is_action_pressed("attack") else C_CYAN
	var radius := 10.0 if Input.is_action_pressed("attack") else 14.0
	draw_circle(mouse, radius, Color(color, 0.7), false, 1.0)
	for direction in [Vector2.RIGHT, Vector2.DOWN, Vector2.LEFT, Vector2.UP]:
		draw_line(mouse + direction * (radius + 3.0), mouse + direction * (radius + 9.0), Color(color, 0.8), 1.0)
