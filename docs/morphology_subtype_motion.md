# Subtype graph-driven motion audit

`forge.morphology_subtype_motion` extends the graph-driven pixel animation
authority from one representative per family to all twenty explicit subtype
chassis. It generates no flipbook substitute and changes no categorical source
pixel. Existing semantic layers, attachment roots, sockets, palettes, loop
endpoints, and gameplay events remain authoritative.

The bank covers all thirteen programs for each subtype in the north-facing
principal view (260 strict clips), plus eight-way locomotion for each subtype
(160 strict directional clips). Every subtype must demonstrate:

- breathing motion in the head, paired arms, and auxiliary appendage;
- locomotor motion in the body, both arms, both legs, and auxiliary appendage;
- a strongly articulated weapon-side attack;
- bilateral arm and auxiliary motion during casting;
- at least eleven distinct peak poses among the thirteen programs;
- at least six distinct directional locomotion signatures.

This is specifically an anti-collapse authority. A clip cannot pass merely by
having the right frame count: per-layer semantic frame diversity and action
excursion are recorded and gated.

```powershell
python -m forge.morphology_subtype_motion build
python -m forge.morphology_subtype_motion validate `
  outputs/morphology_subtype_motion_v1/morphology_subtype_motion.json
```

The contact sheet displays the event/peak pose for all 20 x 13 combinations at
native 48px nearest-neighbor resolution. Validation regenerates every clip and
requires byte-identical report and image closure.
