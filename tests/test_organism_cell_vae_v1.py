from __future__ import annotations
import torch
from forge.organism_cell_vae_v1.contract import CELL_FEATURES,MAX_CELLS,Plan
from forge.organism_cell_vae_v1.model import ContinuousCellVAE,loss
def test_continuous_cell_vae_has_finite_gradient()->None:
    model=ContinuousCellVAE(width=32,latent_dim=12);features=torch.zeros(1,MAX_CELLS,CELL_FEATURES);features[0,:4,:2]=torch.tensor(((-.1,-.1),(.1,-.1),(-.1,.1),(.1,.1)));features[0,:4,2]=1;features[0,:4,51]=1;mask=torch.zeros(1,MAX_CELLS,dtype=torch.bool);mask[:,:4]=True;target=torch.zeros(1,4,96,96);target[:,:,45:52,45:52]=1;output=model(features,mask,stochastic=False);value,metrics=loss(output,target,mask,1);value.backward();assert output.rgba.shape==(1,4,96,96) and torch.isfinite(value) and model.decoder[-1].weight.grad is not None and set(metrics)>={"alpha_bce","dice","kl"}
def test_plan_is_segmented()->None:assert Plan().total_steps//Plan().segment_steps==6
