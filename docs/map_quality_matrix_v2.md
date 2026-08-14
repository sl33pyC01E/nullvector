# Topology-v2 map quality matrix

The compact authoritative breadth bank lives at
`outputs/maps_quality_matrix_v2`. It contains 24 deterministic topology-v2
packs: all six themes at four geometry profiles.

| Profile | Dimensions | Objectives | Spawns | Base seed |
|---|---:|---:|---:|---:|
| compact | 32×32 | 2 | 6 | `0x514D415452495801` |
| portrait | 48×64 | 3 | 8 | `0x514D415452495802` |
| landscape | 64×48 | 3 | 8 | `0x514D415452495803` |
| expanded | 72×72 | 4 | 12 | `0x514D415452495804` |

Each profile is generated for `arena`, `rooms`, `caves`, `archipelago`,
`garden`, and `anomaly`. Rebuild additively into a new empty destination using
the following commands; never overwrite the authority bank in place.

```powershell
python -m forge.maps generate --output outputs/maps_quality_matrix_v2_next `
  --themes all --count 1 --seed 0x514D415452495801 --width 32 --height 32 `
  --objectives 2 --spawns 6 --preview-scale 2
python -m forge.maps generate --output outputs/maps_quality_matrix_v2_next `
  --themes all --count 1 --seed 0x514D415452495802 --width 48 --height 64 `
  --objectives 3 --spawns 8 --preview-scale 2
python -m forge.maps generate --output outputs/maps_quality_matrix_v2_next `
  --themes all --count 1 --seed 0x514D415452495803 --width 64 --height 48 `
  --objectives 3 --spawns 8 --preview-scale 2
python -m forge.maps generate --output outputs/maps_quality_matrix_v2_next `
  --themes all --count 1 --seed 0x514D415452495804 --width 72 --height 72 `
  --objectives 4 --spawns 12 --preview-scale 2
python -m forge.maps validate outputs/maps_quality_matrix_v2_next
```

The exact quality audit and visually inspected contact sheet are under
`outputs/map_quality_matrix_v2`. Current evidence:

- 24/24 hard-valid packs and 24 unique semantic identities;
- report SHA256 `17add6f59b10e73262ecc81d3fbdfaeda85874383df371accb017cecd104ed09`;
- showcase manifest SHA256
  `6ab3cf5414849e90df0063330ea4725fb92e58c2eef31e47ac8af7ffa41219f5`;
- maximum geometric detour ratio 1.526316;
- maximum radius-one-safe detour ratio 1.190476;
- zero minimum-hazard cost across every shortest mission route;
- 4–6 elevation levels in every map;
- agent-scale mission articulation fraction 0–0.01995, retained as a quality
  signal rather than misrepresented as a hard-validity failure.

Re-audit a rebuilt bank with:

```powershell
$packs = Get-ChildItem outputs/maps_quality_matrix_v2_next -Directory |
  Sort-Object Name | ForEach-Object FullName
python -m forge.map_quality @packs `
  --report outputs/map_quality_matrix_v2_next/quality_report.json `
  --showcase outputs/map_quality_matrix_v2_next/showcase
```

The quality report must be checked with `assert_exact_audit_replay`, and the
showcase with `assert_exact_quality_showcase`; checksum validity alone is not
treated as source replay.
