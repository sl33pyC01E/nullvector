# Recurrent world student V1

This is the first reverse-distillation student built after the factorized neural
teacher became loadable. It consumes the current and previous world latents,
action, controls, compact world state, and actor physiology. A convolutional
GRU carries memory across transitions and predicts the next visual latent plus
actor state.

V1 is deliberately small and segmented. It must beat persistence and wrong
actions on an unseen world before it can become a runtime authority. Later
versions add free-rollout decoded-frame gates and distill the remaining
specialist states into the recurrent hidden state.
