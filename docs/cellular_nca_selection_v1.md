# Causal cellular NCA selection

This stage compares the raw and exponential-moving-average states from the frozen V3 physiology run. It evaluates 32-step general injury rollouts and 16-step heart, lung, gut, and neural ablations. A compact runtime model is exported only when one candidate passes every gate.

This selector exists because a short fine-tune can improve the raw model while a slow EMA remains biased toward its parent. Selection is evidence-based; neither state is preferred by name.
