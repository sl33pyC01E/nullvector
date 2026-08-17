# Recurrent Action-DiT v2

This student starts from the promoted 39.5M-parameter Action-DiT and adds two causal inputs the original model lacked: the previous visual latent and the 128-value actor state. Both adapters initialize to zero, so training begins as an exact copy of the accepted parent rather than an unrelated tiny model.

The production corpus contains 1,512 active or settling transitions across six held-out worlds. Training is segmented every 250 updates, validation chooses raw or EMA weights plus a conservative residual gate, and the untouched sixth world decides promotion. A runtime is accepted only if it beats persistence and the correct action beats a deliberately wrong action.
