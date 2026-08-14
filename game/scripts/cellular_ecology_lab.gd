extends "res://scripts/cellular_motion_lab.gd"

const ECOLOGY_FORMAT := "nullvector-cellular-ecology-native-catalog-v1"
const ECOLOGY_ORIGIN := Vector2(505.0, 99.0)
const ECOLOGY_CELL := 12.0
const ECOLOGY_COLORS := [Color("#3fe0ff"), Color("#ff8746"), Color("#6eff56"), Color("#db53ff"), Color("#aabfe1")]

@export_file("*.json") var ecology_catalog_path := "res://generated/cellular_ecology/v1/ecology_catalog.json"

var ecology_catalog: Dictionary = {}
var ecology_map_index := 0
var ecology_label: Label
var habitat_label: Label
var ecology_damage_accumulator := 0.0


func _ready() -> void:
	ecology_catalog = _load_json(ecology_catalog_path)
	_validate_ecology_catalog()
	super()
	_build_ecology_overlay()
	_reset_ecology_resources()
	_refresh_ecology_overlay()
	if not startup_errors.is_empty():
		status_label.text = "FAIL-CLOSED // " + ", ".join(startup_errors)
		status_label.modulate = Color("#ff526d")
	if "--cellular-ecology-smoke" in OS.get_cmdline_user_args(): call_deferred("_run_ecology_smoke")


func _validate_ecology_catalog() -> void:
	if ecology_catalog.get("format", "") != ECOLOGY_FORMAT: startup_errors.append("ecology format")
	if ecology_catalog.get("status", "") != "ready": startup_errors.append("ecology status")
	if int(ecology_catalog.get("map_count", -1)) != 6: startup_errors.append("ecology map census")
	if int(ecology_catalog.get("resource_node_count", -1)) != 120: startup_errors.append("ecology resource census")
	if ecology_catalog.get("maps", []).size() != 6: startup_errors.append("ecology map records")
	for map_record in ecology_catalog.get("maps", []):
		if map_record.get("fields_u8", {}).size() != 7: startup_errors.append("ecology fields")
		if map_record.get("family_suitability_u8", []).size() != 5: startup_errors.append("ecology niches")
		if map_record.get("resource_nodes", []).size() != 20: startup_errors.append("ecology nodes")


func _build_ecology_overlay() -> void:
	var canvas := CanvasLayer.new(); add_child(canvas)
	var panel := _panel(canvas, Rect2(850, 174, 384, 78))
	ecology_label = _label(panel, Vector2(12, 8), Vector2(360, 22), "ECOLOGY", LIME, 10)
	habitat_label = _label(panel, Vector2(12, 31), Vector2(360, 39), "HABITAT", MUTED, 8)
	controls_label.text += "\nA/D habitat // ecological food regrows"


func _current_ecology() -> Dictionary:
	var maps: Array = ecology_catalog.get("maps", [])
	return maps[ecology_map_index] if ecology_map_index >= 0 and ecology_map_index < maps.size() else {}


func _grid_to_world(point: Array) -> Vector2:
	return ECOLOGY_ORIGIN + Vector2(float(point[0]) + 0.5, float(point[1]) + 0.5) * ECOLOGY_CELL


func _world_to_index(position: Vector2) -> int:
	var local := (position - ECOLOGY_ORIGIN) / ECOLOGY_CELL
	var x: int = clampi(int(floor(local.x)), 0, 47); var y: int = clampi(int(floor(local.y)), 0, 47)
	return y * 48 + x


func _field(name: String, position: Vector2) -> float:
	var values: Array = _current_ecology().get("fields_u8", {}).get(name, [])
	var index := _world_to_index(position)
	return float(values[index]) / 255.0 if index >= 0 and index < values.size() else 0.0


func _reset_ecology_resources() -> void:
	foods = foods.filter(func(food: Dictionary): return not food.has("ecology_id"))
	for node in _current_ecology().get("resource_nodes", []):
		foods.append({"position": _grid_to_world(node.get("position", [0, 0])), "amount": float(node.get("capacity", 20.0)), "capacity": float(node.get("capacity", 20.0)), "regrowth": float(node.get("regrowth_per_second", 0.1)), "pulse": 0.0, "ecology_id": int(node.get("id", -1)), "family_id": int(node.get("family_id", 0))})
	queue_redraw()


func _step_food(delta: float) -> void:
	super(delta)
	var present: Dictionary = {}
	for food in foods:
		if food.has("ecology_id"):
			present[str(int(food["ecology_id"]))] = true
			food["amount"] = minf(float(food.get("capacity", 20.0)), float(food["amount"]) + float(food.get("regrowth", 0.1)) * delta)
	for node in _current_ecology().get("resource_nodes", []):
		var key := str(int(node.get("id", -1)))
		if not present.has(key):
			foods.append({"position": _grid_to_world(node.get("position", [0, 0])), "amount": float(node.get("regrowth_per_second", 0.1)) * delta, "capacity": float(node.get("capacity", 20.0)), "regrowth": float(node.get("regrowth_per_second", 0.1)), "pulse": 0.0, "ecology_id": int(node.get("id", -1)), "family_id": int(node.get("family_id", 0))})


func _step_organism(organism: Dictionary, delta: float) -> void:
	super(organism, delta)
	var center := _organism_center(organism); var toxicity := _field("toxicity", center); var temperature := _field("temperature", center)
	var stress: float = maxf(0.0, toxicity - 0.48) + maxf(0.0, absf(temperature - 0.52) - 0.32)
	if stress > 0.0:
		ecology_damage_accumulator += stress * delta
		for index in range(organism["alive"].size()):
			if organism["alive"][index]: organism["health"][index] = maxf(0.0, float(organism["health"][index]) - stress * delta * 0.016)


func _can_reproduce(organism: Dictionary) -> bool:
	if not super(organism): return false
	var center := _organism_center(organism); var biomass := _field("biomass", center)
	var carrying_capacity := 1 + int(floor(biomass * 7.0))
	return biomass >= 0.18 and organisms.size() < carrying_capacity


func _change_ecology(delta: int) -> void:
	var maps: Array = ecology_catalog.get("maps", [])
	if maps.is_empty(): return
	ecology_map_index = posmod(ecology_map_index + delta, maps.size()); _reset_ecology_resources(); _refresh_ecology_overlay(); queue_redraw()


func _refresh_ecology_overlay() -> void:
	if ecology_label == null: return
	var record := _current_ecology(); ecology_label.text = "HABITAT // " + str(record.get("theme", "?")).to_upper()
	var stats: Dictionary = record.get("statistics", {})
	habitat_label.text = "NUTRIENT %.2f  MOISTURE %.2f  LIGHT %.2f\nTOXICITY %.2f  ENERGY %.2f  BIOMASS %.2f" % [float(stats.get("nutrient", {}).get("mean", 0.0)), float(stats.get("moisture", {}).get("mean", 0.0)), float(stats.get("light", {}).get("mean", 0.0)), float(stats.get("toxicity", {}).get("mean", 0.0)), float(stats.get("energy", {}).get("mean", 0.0)), float(stats.get("biomass", {}).get("mean", 0.0))]


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_A: _change_ecology(-1); get_viewport().set_input_as_handled(); return
		if event.keycode == KEY_D: _change_ecology(1); get_viewport().set_input_as_handled(); return
	super(event)


func _draw() -> void:
	super()
	var record := _current_ecology(); var fields: Dictionary = record.get("fields_u8", {})
	var biomass: Array = fields.get("biomass", []); var moisture: Array = fields.get("moisture", []); var toxicity: Array = fields.get("toxicity", [])
	if biomass.size() == 2304:
		for y in range(48):
			for x in range(48):
				var index := y * 48 + x; var color := Color(float(toxicity[index]) / 255.0 * 0.45, float(biomass[index]) / 255.0 * 0.52, float(moisture[index]) / 255.0 * 0.60, 0.20)
				draw_rect(Rect2(ECOLOGY_ORIGIN + Vector2(x, y) * ECOLOGY_CELL, Vector2(ECOLOGY_CELL, ECOLOGY_CELL)), color, true)
		for organism in organisms: _draw_organism(organism)
	for node in record.get("resource_nodes", []):
		var color: Color = ECOLOGY_COLORS[int(node.get("family_id", 0))]; var position := _grid_to_world(node.get("position", [0, 0]))
		draw_circle(position, 5.5, Color(color.r, color.g, color.b, 0.22)); draw_circle(position, 2.2, color)
	draw_rect(Rect2(ECOLOGY_ORIGIN, Vector2(48, 48) * ECOLOGY_CELL), Color("#2b5970"), false, 1.0)


func _run_ecology_smoke() -> void:
	var errors: Array[String] = startup_errors.duplicate(); var resource_count := 0; var field_cells := 0
	for record in ecology_catalog.get("maps", []):
		resource_count += record.get("resource_nodes", []).size()
		for values in record.get("fields_u8", {}).values(): field_cells += values.size()
	if resource_count != 120: errors.append("resource total")
	if field_cells != 6 * 7 * 48 * 48: errors.append("field cell total")
	if foods.size() < 20: errors.append("live resource population")
	var before := float(foods[0].get("amount", 0.0)); foods[0]["amount"] = maxf(0.0, before - 5.0); _step_food(1.0)
	if float(foods[0].get("amount", 0.0)) <= before - 5.0: errors.append("resource regrowth")
	var report := {"format": "nullvector-cellular-ecology-godot-smoke-v1", "passed": errors.is_empty(), "errors": errors, "engine": Engine.get_version_info().get("string", ""), "ecology_bundle_id": ecology_catalog.get("bundle_id", ""), "motion_bundle_id": motion_catalog.get("bundle_id", ""), "organism_bundle_id": catalog.get("bundle_id", ""), "map_count": ecology_catalog.get("map_count", 0), "resource_node_count": resource_count, "field_cell_count": field_cells, "organism_count": organisms.size(), "ecological_damage_accumulator": ecology_damage_accumulator, "python_runtime_required": false}
	var report_path := "res://../outputs/cellular_ecology_godot_report.json"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--cellular-ecology-report="): report_path = argument.trim_prefix("--cellular-ecology-report=")
	var file := FileAccess.open(report_path, FileAccess.WRITE)
	if file != null: file.store_string(JSON.stringify(report, "  ", false)); file.close()
	print("CELLULAR_ECOLOGY_SMOKE_%s maps=%d resources=%d fields=%d" % ["OK" if errors.is_empty() else "FAIL", int(report["map_count"]), resource_count, field_cells])
	get_tree().quit(0 if errors.is_empty() else 1)
