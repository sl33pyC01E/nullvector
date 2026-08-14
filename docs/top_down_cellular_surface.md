# Top-down cellular surface contract

The cellular simulation uses one explicit camera/physics interpretation:
`top_down_dorsal`. Screen X and screen Y are both axes of the world surface.
Neither axis represents vertical height.

The hash-bound runtime contract lives in
`forge/cellular_organism/orientation.py` and is projected into every base,
evolved, and structurally bred native catalog. Godot fails closed when the
contract is absent or differs.

## Consequences

- Uniform screen-down gravity is disabled and the Python reference rejects
  `step(..., gravity=True)`.
- Living motion is planar organ actuation.
- Dead or separated tissue keeps trauma momentum and receives only weak planar
  attraction toward viable tissue. This is the reconnection tendency, not
  gravity.
- Internal fluid still moves through intact bonds by pressure.
- Escaped fluid is deposited on the world surface. A symmetric 3x3 kernel and
  eight-neighbor diffusion expand the puddle without an X/Y directional bias.
- Godot renders puddles as expanding pixel lobes. Directional streaking can
  arise from wound/body momentum, never from an implicit screen-down force.
- Dorsal breathing is radial. Locomotion may travel along both screen axes.

The anatomical bank remains immutable. Orientation is a runtime contract, so
the same categorical cells, organs, genomes, and breakable bonds can be reused
without rewriting biological identity.

Historical anatomy-bank metadata may still record scalar gravity `28`; that
value is archival provenance only. Every active native catalog projects it to
`0` and binds this top-down contract before Godot can load it.

## Current native evidence

The three current native catalogs are repeat-exact:

| Runtime | Species | Cells | Bonds | Bundle ID |
| --- | ---: | ---: | ---: | --- |
| base organisms | 80 | 34,178 | 116,112 | `1e666c02a24b0d46a56d708d1a3a95fa8d7705d293b6555666dd77ec1d60ca1b` |
| evolved descendants | 36 | 14,457 | 48,569 | `a2471c7739263ee1178282f1c574a2c9ee97e31ec81664bb643a58ffd941ee76` |
| structural offspring | 45 | 22,933 | 77,829 | `b956f84de997f6e805e0aa3113ac6d353264b7c046c202d52e47d5e1168fa3dc` |

Headless Godot reports:

- `outputs/cellular_organism_topdown_godot_report.json`
- `outputs/evolved_cellular_organism_topdown_godot_report.json`
- `outputs/cellular_breeding_topdown_godot_report.json`

Each report proves catalog loading, trauma, leakage, surface-puddle creation,
feeding, reproduction, the top-down projection, and the absence of uniform
screen gravity.

## Reference validation

```powershell
C:\Users\forre\AppData\Local\Programs\Python\Python312\python.exe -m pytest `
  tests/test_cellular_surface.py `
  tests/test_cellular_organism.py `
  tests/test_cellular_organism_native.py `
  tests/test_evolved_cellular_organism_native.py `
  tests/test_cellular_breeding_native.py -q
```

`test_cellular_surface.py` deposits fluid at the center of a square lattice,
advances 180 updates, and requires exact horizontal, vertical, and transpose
symmetry while the RMS radius grows and mass remains conserved.
