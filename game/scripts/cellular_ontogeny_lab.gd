extends "res://scripts/cellular_ecology_lab.gd"

const ONTOGENY_FORMAT := "nullvector-cellular-ontogeny-native-catalog-v1"
const STAGE_NAMES := ["ZYGOTE", "GASTRULA", "ORGAN PRIMORDIA", "LARVAL", "JUVENILE", "ADULT"]
const LINEAGE_COLORS := [Color("#000000"), Color("#4ad5ff"), Color("#ff638f"), Color("#ffb744"), Color("#c764ff"), Color("#68ff7e")]

@export_file("*.json") var ontogeny_catalog_path := "res://generated/cellular_ontogeny/v1/ontogeny_catalog.json"

var ontogeny_catalog: Dictionary = {}
var ontogeny_programs: Dictionary = {}
var ontogeny_label: Label
var lineage_label: Label
var growth_speed := 1.0


func _ready() -> void:
	ontogeny_catalog = _load_json(ontogeny_catalog_path); _validate_ontogeny_catalog()
	super(); _cross_validate_ontogeny(); _build_ontogeny_overlay(); _refresh_ontogeny_overlay()
	if not startup_errors.is_empty(): status_label.text = "FAIL-CLOSED // " + ", ".join(startup_errors); status_label.modulate = Color("#ff526d")
	if "--cellular-ontogeny-smoke" in OS.get_cmdline_user_args(): call_deferred("_run_ontogeny_smoke")


func _validate_ontogeny_catalog() -> void:
	if ontogeny_catalog.get("format", "") != ONTOGENY_FORMAT: startup_errors.append("ontogeny format")
	if ontogeny_catalog.get("status", "") != "ready": startup_errors.append("ontogeny status")
	if int(ontogeny_catalog.get("program_count", -1)) != 45: startup_errors.append("ontogeny census")
	for program in ontogeny_catalog.get("programs", []): ontogeny_programs[str(program.get("sample_id", ""))] = program
	if ontogeny_programs.size() != 45: startup_errors.append("ontogeny identity census")


func _cross_validate_ontogeny() -> void:
	for entry in catalog.get("species", []):
		var sample_id := str(entry.get("sample_id", "")); var program: Dictionary = ontogeny_programs.get(sample_id, {})
		if program.is_empty(): startup_errors.append("ontogeny identity " + sample_id); continue
		if program.get("birth_order", []).size() != int(entry.get("summary", {}).get("physical_cell_count", -1)): startup_errors.append("ontogeny cells " + sample_id)
		if program.get("bond_activation_stage", []).size() != int(entry.get("summary", {}).get("bond_count", -1)): startup_errors.append("ontogeny bonds " + sample_id)


func _build_ontogeny_overlay() -> void:
	var canvas := CanvasLayer.new(); add_child(canvas); var panel := _panel(canvas, Rect2(850, 174, 384, 78))
	ontogeny_label = _label(panel, Vector2(12, 8), Vector2(360, 22), "ONTOGENY", LIME, 10)
	lineage_label = _label(panel, Vector2(12, 31), Vector2(360, 39), "LINEAGES", MUTED, 8)
	controls_label.text += "\nG advance stage  Shift+G adult  X growth speed"


func _create_organism(data: Dictionary, center: Vector2, generation: int, mutation_seed: int) -> Dictionary:
	var organism: Dictionary = super(data, center, generation, mutation_seed)
	if organism.is_empty(): return organism
	var program: Dictionary = ontogeny_programs.get(str(data.get("sample_id", "")), {})
	if program.is_empty(): return organism
	organism["ontogeny_program"] = program; organism["development_age"] = 0.0; organism["development_progress"] = 0.0; organism["development_stage"] = 0
	organism["development_target_health"] = organism["health"].duplicate(); organism["development_target_fluid"] = organism["fluid"].duplicate(); organism["development_target_energy"] = organism["energy"].duplicate(); organism["development_target_nutrient"] = organism["nutrient"].duplicate()
	for index in range(organism["alive"].size()): organism["alive"][index] = false; organism["health"][index] = 0.0; organism["fluid"][index] = 0.0; organism["energy"][index] = 0.0; organism["nutrient"][index] = 0.0
	for index in range(organism["bond_alive"].size()): organism["bond_alive"][index] = false
	_activate_to_rank(organism, _stage_cell_count(program, 0)); return organism


func _stage_cell_count(program: Dictionary, stage: int) -> int:
	var stages: Array = program.get("stages", []); return int(stages[clampi(stage, 0, stages.size() - 1)].get("cell_count", 1)) if not stages.is_empty() else 1


func _activate_to_rank(organism: Dictionary, target_count: int) -> void:
	var program: Dictionary = organism["ontogeny_program"]; var order: Array = program.get("birth_order", []); var parents: Array = program.get("parent_cell", [])
	var by_rank: Array = []; by_rank.resize(order.size())
	for cell in range(order.size()): by_rank[int(order[cell])] = cell
	for rank in range(mini(target_count, by_rank.size())):
		var cell := int(by_rank[rank])
		if organism["alive"][cell]: continue
		var parent := int(parents[cell]); organism["alive"][cell] = true
		if parent >= 0 and organism["alive"][parent]: organism["position"][cell] = organism["position"][parent] + Vector2.RIGHT.rotated(float(cell) * 2.39996) * 0.8
		organism["health"][cell] = float(organism["development_target_health"][cell]) * 0.55; organism["fluid"][cell] = float(organism["development_target_fluid"][cell]) * 0.45; organism["energy"][cell] = float(organism["development_target_energy"][cell]) * 0.65; organism["nutrient"][cell] = float(organism["development_target_nutrient"][cell]) * 0.50
	var bond_stage: Array = program.get("bond_activation_stage", []); var active_stage := 0
	for stage in range(6):
		if target_count >= _stage_cell_count(program, stage): active_stage = stage
	for bond_index in range(organism["bond_alive"].size()):
		var pair: Array = organism["bond_ab"][bond_index]
		if int(bond_stage[bond_index]) <= active_stage and organism["alive"][int(pair[0])] and organism["alive"][int(pair[1])]: organism["bond_alive"][bond_index] = true


func _advance_development(organism: Dictionary, delta: float) -> void:
	if not organism.has("ontogeny_program") or float(organism["development_progress"]) >= 1.0: return
	organism["development_age"] = float(organism["development_age"]) + delta * growth_speed
	var duration := maxf(0.1, float(organism["ontogeny_program"].get("development_seconds", 20.0))); var progress: float = clampf(float(organism["development_age"]) / duration, 0.0, 1.0)
	organism["development_progress"] = progress; var total: int = organism["alive"].size(); var zygote_count: int = _stage_cell_count(organism["ontogeny_program"], 0)
	var target_count: int = mini(total, maxi(zygote_count, int(ceil(progress * total)))); _activate_to_rank(organism, target_count)
	var stage := 0
	for candidate in range(6):
		if target_count >= _stage_cell_count(organism["ontogeny_program"], candidate): stage = candidate
	if stage != int(organism["development_stage"]): organism["development_stage"] = stage; _event("ONTOGENY // " + STAGE_NAMES[stage], LIME)
	if progress >= 1.0:
		for index in range(organism["alive"].size()):
			if organism["alive"][index]: organism["health"][index] = maxf(float(organism["health"][index]), float(organism["development_target_health"][index])); organism["fluid"][index] = maxf(float(organism["fluid"][index]), float(organism["development_target_fluid"][index]))


func _step_organism(organism: Dictionary, delta: float) -> void:
	_advance_development(organism, delta); super(organism, delta)


func _can_reproduce(organism: Dictionary) -> bool:
	return float(organism.get("development_progress", 1.0)) >= 1.0 and super(organism)


func _cell_color(organism: Dictionary, index: int) -> Color:
	var base := super(organism, index)
	if organism.has("ontogeny_program"):
		var lineage := int(organism["ontogeny_program"].get("lineage_id", [1])[index]); base = base.lerp(LINEAGE_COLORS[lineage], 0.28)
	return base


func _force_stage(adult: bool) -> void:
	if organisms.is_empty() or not organisms[0].has("ontogeny_program"): return
	var organism := organisms[0]; var next_stage := 5 if adult else mini(5, int(organism["development_stage"]) + 1); var target := _stage_cell_count(organism["ontogeny_program"], next_stage)
	_activate_to_rank(organism, target); organism["development_stage"] = next_stage
	if adult: organism["development_age"] = float(organism["ontogeny_program"].get("development_seconds", 20.0)); organism["development_progress"] = 1.0
	else: organism["development_progress"] = float(target) / maxi(1, organism["alive"].size()); organism["development_age"] = float(organism["development_progress"]) * float(organism["ontogeny_program"].get("development_seconds", 20.0))
	_event("ONTOGENY // " + STAGE_NAMES[next_stage], LIME); _refresh_ontogeny_overlay(); queue_redraw()


func _refresh_ontogeny_overlay() -> void:
	if ontogeny_label == null or organisms.is_empty(): return
	var organism := organisms[0]; var stage := int(organism.get("development_stage", 0)); var active: int = organism["alive"].count(true)
	ontogeny_label.text = "DEVELOPMENT // " + STAGE_NAMES[stage] + " // %d%%" % int(float(organism.get("development_progress", 0.0)) * 100.0)
	lineage_label.text = "ACTIVE %d / %d CELLS // SPEED %.1fx\nECTO / MESO / ENDO / GERMLINE / SPECIALIZED" % [active, organism["alive"].size(), growth_speed]


func _physics_process(delta: float) -> void:
	super(delta); _refresh_ontogeny_overlay()


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		if event.keycode == KEY_G: _force_stage(event.shift_pressed); get_viewport().set_input_as_handled(); return
		if event.keycode == KEY_X: growth_speed = 4.0 if growth_speed == 1.0 else (0.0 if growth_speed == 4.0 else 1.0); _refresh_ontogeny_overlay(); get_viewport().set_input_as_handled(); return
	super(event)


func _run_ontogeny_smoke() -> void:
	var errors: Array[String] = startup_errors.duplicate(); var cells := 0; var bonds := 0
	for program in ontogeny_catalog.get("programs", []): cells += program.get("birth_order", []).size(); bonds += program.get("bond_activation_stage", []).size()
	if cells != int(ontogeny_catalog.get("totals", {}).get("cells", -1)): errors.append("ontogeny cell total")
	if bonds != int(ontogeny_catalog.get("totals", {}).get("bonds", -1)): errors.append("ontogeny bond total")
	if organisms.is_empty(): errors.append("zygote spawn")
	else:
		var organism := organisms[0]; var initial: int = organism["alive"].count(true); _force_stage(true); var adult: int = organism["alive"].count(true)
		if initial < 1 or initial >= adult: errors.append("development growth")
		if adult != organism["alive"].size(): errors.append("adult cell exactness")
	var report := {"format": "nullvector-cellular-ontogeny-godot-smoke-v1", "passed": errors.is_empty(), "errors": errors, "engine": Engine.get_version_info().get("string", ""), "ontogeny_bundle_id": ontogeny_catalog.get("bundle_id", ""), "motion_bundle_id": motion_catalog.get("bundle_id", ""), "organism_bundle_id": catalog.get("bundle_id", ""), "program_count": ontogeny_catalog.get("program_count", 0), "cell_count": cells, "bond_count": bonds, "zygote_cell_count": organisms[0]["ontogeny_program"].get("stages", [])[0].get("cell_count", 0) if not organisms.is_empty() else 0, "adult_cell_count": organisms[0]["alive"].count(true) if not organisms.is_empty() else 0, "python_runtime_required": false}
	var report_path := "res://../outputs/cellular_ontogeny_godot_report.json"
	for argument in OS.get_cmdline_user_args():
		if argument.begins_with("--cellular-ontogeny-report="): report_path = argument.trim_prefix("--cellular-ontogeny-report=")
	var file := FileAccess.open(report_path, FileAccess.WRITE)
	if file != null: file.store_string(JSON.stringify(report, "  ", false)); file.close()
	print("CELLULAR_ONTOGENY_SMOKE_%s programs=%d cells=%d bonds=%d" % ["OK" if errors.is_empty() else "FAIL", int(report["program_count"]), cells, bonds]); get_tree().quit(0 if errors.is_empty() else 1)
