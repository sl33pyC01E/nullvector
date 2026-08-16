# Anatomical graph rasterizer

This successor turns the organism raster model into an interface for the game,
not merely an image compressor. Its conditioning sequence contains stable
appendage, articulated-joint, and organ tokens. Every visible logical cell has
independent appendage, joint, and organ authority targets where applicable.
These targets use group-normalized hierarchical distributions: a joint does not
compete with its parent appendage, and an organ does not erase either. A cell
may legitimately carry all three ownership levels.

The target renderer is unchanged from VAE v3/v4. That makes held-out visual and
motion comparisons honest: improvement must come from the anatomical graph,
not from replacing the reference images.

The authority maps are intentionally reusable outside rendering:

- joint and appendage ownership define severing, grafting, planted contacts,
  muscle actuation, and locomotor impairment;
- organ ownership defines metabolism, respiration, circulation, sensing,
  cognition, healing, and death cascades;
- cell-level ownership lets the powder simulation damage real subsystems rather
  than subtracting an abstract hit-point total.

This is still a scaffolded specialist model. It is not the final monolithic
action-conditioned world model and must not be described as the complete game.
