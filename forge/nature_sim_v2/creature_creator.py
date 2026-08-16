from __future__ import annotations

from dataclasses import dataclass,field,replace
import hashlib

from ..creature_stage_developmental import FAMILIES,develop
from .contract import EcoGenome
from .directed_evolution import apply_offer,evolution_offers
from .genetics import founder_genomes
from .grafting import graft_appendage_pair,graft_organ


@dataclass(slots=True)
class CreatureCreator:
    active:bool=False
    family:int=0
    variant:int=0
    donor_family:int=1
    graft_kind:str="none"
    offer_epoch:int=0
    selected_offers:set[int]=field(default_factory=set)
    revision:int=0

    def change(self,*,family:int=0,variant:int=0,donor:int=0)->None:
        self.family=(self.family+family)%len(FAMILIES);self.variant=(self.variant+variant)%6;self.donor_family=(self.donor_family+donor)%len(FAMILIES);self.selected_offers.clear();self.revision+=1

    def cycle_graft(self)->None:
        self.graft_kind=("none","organ","locomotor")[(("none","organ","locomotor").index(self.graft_kind)+1)%3];self.revision+=1

    def cycle_offer_epoch(self)->None:self.offer_epoch+=1;self.selected_offers.clear();self.revision+=1

    def toggle_offer(self,index:int)->None:
        if not 0<=index<3:raise ValueError("creator offer index drifted")
        if index in self.selected_offers:self.selected_offers.remove(index)
        elif len(self.selected_offers)<3:self.selected_offers.add(index)
        self.revision+=1

    @property
    def base(self)->EcoGenome:return founder_genomes(variants_per_family=6)[self.family*6+self.variant]

    @property
    def offers(self):return evolution_offers(self.base,epoch=self.offer_epoch)

    def cost(self)->dict[str,float]:
        result={"biomass":1.5+.45*len(self.selected_offers),"knowledge":.35+.35*len(self.selected_offers)}
        if self.graft_kind=="organ":result["crystal"]=.6
        elif self.graft_kind=="locomotor":result["metal"]=.75
        return result

    def genome(self,*,seed:int)->EcoGenome:
        genome=self.base
        for index in sorted(self.selected_offers):genome=apply_offer(genome,self.offers[index],seed=(seed+index*7919)&0x7FFF_FFFF_FFFF_FFFF)
        if self.graft_kind!="none" and self.donor_family!=self.family:
            donor=founder_genomes(variants_per_family=6)[self.donor_family*6+self.variant];rng_seed=(seed^0x4752414654)&0x7FFF_FFFF_FFFF_FFFF
            if self.graft_kind=="organ":
                candidates=[item for item in donor.developmental.components if item.organ!="none"]
                if candidates:genome=graft_organ(genome,donor,candidates[rng_seed%len(candidates)].component_id,seed=rng_seed)
            else:
                candidates=[item for item in donor.developmental.appendages if item.paired_with is not None]
                for item in candidates:
                    try:genome=graft_appendage_pair(genome,donor,item.appendage_id,seed=rng_seed);break
                    except ValueError:continue
        identity=hashlib.sha256(f"{genome.semantic_sha256()}:{seed}:created".encode()).hexdigest()[:12]
        return EcoGenome(replace(genome.developmental,genome_id=f"created_{identity}",seed=int(seed),parent_ids=(genome.developmental.genome_id,)),genome.eco_traits,genome.diet,f"created-{FAMILIES[self.family]}-{identity}",genome.mutation_log+("field_incarnation",))

    def incarnate(self,world,adventure,position,*,seed:int)->int:
        cost=self.cost();missing=[name for name,amount in cost.items() if adventure.inventory.get(name,0)+1e-9<amount]
        if missing:raise ValueError("needs "+", ".join(missing))
        genome=self.genome(seed=seed);develop(genome.developmental)
        for name,amount in cost.items():adventure.inventory[name]-=amount
        entity_id=world.add_organism(genome,tuple(position),energy=.72);entity=world.organisms[entity_id];entity.age=8+genome.trait("maturity")*18;entity.reserve=.32;entity.update_stage();world.events.append({"tick":world.tick_index,"type":"incarnation","entity":entity_id,"family":self.family,"variant":self.variant,"graft":self.graft_kind,"offers":sorted(self.selected_offers)});return entity_id
