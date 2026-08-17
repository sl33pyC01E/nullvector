# Neural action-frame compositor

The recurrent Action-DiT predicts the next latent state. The adapted VAE decoder turns its predicted change into pixels. The compositor applies that decoded change to the exact current frame, so static pixels do not inherit codec reconstruction error.

The fifth cellular world selects one residual scale. The sixth world remains the test. Promotion requires lower frame MAE than exact frame persistence and a lower error for the requested action than for a deliberately wrong action.
