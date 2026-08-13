# Neural sprite quality audit

`forge.sprite_quality` is an independent, read-only audit over the immutable
production neural sprite banks. It deliberately separates integrity gates from
visual-quality diagnostics.

The audit verifies all 80 static neural samples and their seven derived layers,
recomputes categorical and presentation metrics, checks family/subtype/role
coverage, measures categorical and silhouette diversity, and validates every
pixel cell in the five-identity motion bank. Motion diagnostics cover all 520
clips and 4,720 stored frames, with duplicated loop endpoints excluded from
motion-energy calculations.

The hard gates prove source binding, deterministic replay, exact categorical
authority, complete matrix coverage, loop coherence, and absence of collapsed
motion clips. They do **not** prove that a sprite is aesthetically pleasing or
conventionally readable. The current humanoid and machine representatives are
explicitly recorded as stylized/abstract, and the motion bank contains five
actual neural representatives rather than all 80 identities.

Run the immutable production audit:

```powershell
python -m forge.sprite_quality --output outputs/sprite_quality/production_v1
```

The output contains a canonical JSON report and a deterministic heatmap of mean
composite change for every family/action cell. `assert_exact_sprite_quality_replay`
rebuilds the entire report, revalidates all inputs, and byte-compares the heatmap.
