# Neural world synthesis v1

This stage composes the selected scale-aware topology prior with the accepted
v4 sparse-object decorator. It produces one complete 32×32 region per biome,
plus a runtime PNG/JSON atlas.

The three authorities stay separate:

- the neural prior emits latent topology tokens and decoded terrain, hazard,
  and elevation fields;
- the deterministic topology compiler repairs only safety failures and records
  every edit in a replayable ledger;
- the accepted neural selector places decal and prop classes through the final
  protected-backbone, clearance, hazard, and legality masks.

Raw tokens and categorical fields are retained beside the compiled map packs.
The manifest reports raw reachability and exact repair fraction per biome, so
this output cannot disguise compiler work as neural quality. A region is
rejected if repair exceeds 15% or any decorated field violates the map masks.

World synthesis runs on region creation or in a background worker. It is not in
the display loop: the live target remains 30 FPS with 30 Hz organism physics
and a 15 Hz causal world.

## Native runtime cache

The additive sync step projects a validated synthesis bank into two native
files: one PNG atlas and one canonical JSON catalog. It ships no checkpoint and
requires neither Python nor CUDA during play.

```powershell
python -m forge.neural_world_synthesis_sync `
  --source outputs/neural_world_synthesis_v1/build_004 `
  --destination game/generated/neural_world_synthesis/v1
```

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m forge.neural_world_synthesis_v1 build
python -m forge.neural_world_synthesis_v1 validate
```

The output is experimental until raw radius-one reachability improves enough
to reduce the repair compiler from ordinary authority to a rare safety net.
