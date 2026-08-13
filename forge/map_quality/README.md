# Map quality audit

This package is a read-only design-diagnostic layer over authoritative
topology-v2 map packs. It never changes generation semantics. A pack is scored
only after the normal map validator verifies schema, artifacts, all topology
invariants, and exact seed replay.

The audit reports:

- raw graph articulation cells plus agent-scale hazard-free chokepoints whose
  removal separates the start from an exit or objective;
- geometric, hazard-free, and square-radius-one hazard-free route lengths;
- canonical-path detour, Chebyshev clearance, nearby hazards, and elevation
  changes, explicitly labeled as tie-order dependent;
- intrinsic widest-clearance and minimum-hazard results over every shortest
  path, independent of BFS tie order;
- eroded walkable fractions, spawn dispersion, zone diversity, elevation
  entropy, and protected-backbone clearance;
- descriptive diagnostics for alternate routes, detour, locomotion clearance,
  spawn collapse, and elevation variety.

Diagnostics are deliberately not validity gates: a mission chokepoint or a flat
arena can be intentional. `hard_validity_preserved` means the authoritative map
contract passed, not that every style heuristic is true.

Run an audit over persisted packs:

```powershell
$packs = Get-ChildItem outputs/maps_v2_forge_lab -Directory |
  Sort-Object Name |
  ForEach-Object FullName
python -m forge.map_quality @packs `
  --report outputs/map_quality/forge_lab_v1.json `
  --showcase outputs/map_quality/showcase
```

Reports bind the exact source manifest and actual semantic-array hashes, an
explicit audit dependency/schema/runtime contract, canonical ordering, all
derived aggregates, and nested plus bank checksums. Checksums detect accidental
corruption; `assert_exact_audit_replay` is the authority against a fully
rehashed edit because it re-audits the persisted source packs and exact-compares
the whole result.

The showcase overlays the captured protected route in blue, agent-center
articulations in amber, and mission-disconnecting agent-center cells in red.
Its PNG and manifest also support exact replay.
