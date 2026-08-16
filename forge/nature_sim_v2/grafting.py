from __future__ import annotations

from dataclasses import dataclass,replace
import hashlib
import numpy as np

from ..creature_stage_developmental import AppendageGene,ComponentGene,DevelopmentalGenome,develop
from .contract import EcoGenome


@dataclass(frozen=True,slots=True)
class HarvestedPart:
    harvest_id:str
    donor_genome_sha256:str
    donor_family:int
    kind:str
    source_ids:tuple[str,...]
    viability:float
    nutrient:float
    mineral:float
    inherited_traits:tuple[float,...]


def harvest_appendage_pair(donor:EcoGenome,appendage_id:str,*,damage:float=.1)->HarvestedPart:
    lookup={a.appendage_id:a for a in donor.developmental.appendages};source=lookup[appendage_id]
    pair=(source,) if source.paired_with is None else (source,lookup[source.paired_with])
    ids=tuple(sorted(a.appendage_id for a in pair));digest=hashlib.sha256((donor.semantic_sha256()+":"+":".join(ids)).encode()).hexdigest()
    size=sum(np.linalg.norm(a.endpoint) for a in pair);machine=donor.family==4
    return HarvestedPart(f"h-{digest[:16]}",donor.semantic_sha256(),donor.family,source.kind,ids,max(0,1-damage),float(size*.08*(not machine)),float(size*.06*machine),tuple(np.mean([a.trait_delta for a in pair],axis=0)))


def graft_appendage_pair(recipient:EcoGenome,donor:EcoGenome,appendage_id:str,*,seed:int)->EcoGenome:
    donor_lookup={a.appendage_id:a for a in donor.developmental.appendages};source=donor_lookup[appendage_id]
    pair=(source,) if source.paired_with is None else (source,donor_lookup[source.paired_with])
    if len(recipient.developmental.appendages)+len(pair)>32:raise ValueError("graft exceeds appendage capacity")
    root=recipient.developmental.components[0].component_id;suffix=hashlib.sha256(f"{seed}:{recipient.semantic_sha256()}:{donor.semantic_sha256()}".encode()).hexdigest()[:7];new=[]
    if len(pair)==2:
        ordered=sorted(pair,key=lambda a:a.side);ids=(f"graft_{suffix}_l",f"graft_{suffix}_r")
        for index,item in enumerate(ordered):
            endpoint=np.asarray(item.endpoint,float);root_offset=np.asarray(item.root_offset,float)
            # Map the whole reciprocal locomotor pair onto the recipient soma;
            # never import only a stray fifth leg or tail.
            new.append(replace(item,appendage_id=ids[index],paired_with=ids[1-index],root_component=root,root_offset=(float(np.sign(item.side)*abs(root_offset[0])),float(root_offset[1])),endpoint=(float(np.sign(item.side)*abs(endpoint[0])),float(endpoint[1]))))
    else:
        item=pair[0];new.append(replace(item,appendage_id=f"graft_{suffix}_center",paired_with=None,root_component=root))
    generation=recipient.developmental.generation+1;developmental=replace(recipient.developmental,genome_id=f"graft_g{generation}_{seed:016x}",seed=int(seed),appendages=recipient.developmental.appendages+tuple(new),generation=generation,parent_ids=(recipient.developmental.genome_id,donor.developmental.genome_id));develop(developmental)
    diet=tuple(np.clip(np.asarray(recipient.diet)*.94+np.asarray(donor.diet)*.06,0,1));eco=recipient.eco_traits
    return EcoGenome(developmental,eco,diet,recipient.lineage_id,recipient.mutation_log+(f"graft_appendage:{source.kind}:{donor.family}",))


def graft_organ(recipient:EcoGenome,donor:EcoGenome,component_id:str,*,seed:int)->EcoGenome:
    source=next(c for c in donor.developmental.components if c.component_id==component_id)
    if source.organ=="none":raise ValueError("cannot graft a non-organ component")
    if len(recipient.developmental.components)>=32:raise ValueError("graft exceeds component capacity")
    root=recipient.developmental.components[0];suffix=hashlib.sha256(f"{seed}:{component_id}".encode()).hexdigest()[:7];rng=np.random.default_rng(seed);side=-1 if rng.random()<.5 else 1
    anchor=(float(root.anchor[0]+side*(root.radius[0]*.42+source.radius[0]*.35)),float(root.anchor[1]+rng.uniform(-.25,.25)*root.radius[1]));radius=tuple(np.clip(np.asarray(source.radius)*(.58+.20*rng.random()),.65,6.5));component=replace(source,component_id=f"graft_{suffix}_{component_id}",anchor=anchor,radius=radius,parent=root.component_id,side=side)
    generation=recipient.developmental.generation+1;developmental=replace(recipient.developmental,genome_id=f"organ_graft_g{generation}_{seed:016x}",seed=int(seed),components=recipient.developmental.components+(component,),generation=generation,parent_ids=(recipient.developmental.genome_id,donor.developmental.genome_id));develop(developmental)
    traits=np.asarray(recipient.eco_traits);donor_traits=np.asarray(donor.eco_traits);eco=tuple(np.clip(traits*.97+donor_traits*.03,0,1));return EcoGenome(developmental,eco,recipient.diet,recipient.lineage_id,recipient.mutation_log+(f"graft_organ:{source.organ}:{donor.family}",))
