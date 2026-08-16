from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import numpy as np

from ..creature_stage_developmental import FAMILIES, develop
from ..living_body_substrate import LivingBody
from ..powder_world_v1 import MaterialGrid
from ..powder_world_v1.contract import MATERIALS,STATE
from .contract import ECO_TRAITS, RESOURCE_NAMES, EcoGenome, WorldSnapshot
from .genetics import founder_genomes, recombine
from .grafting import graft_appendage_pair, graft_organ
from .state import ColonyState, OrganismState
from .colony_ecology import ColonyEcology
from .climate import ClimateSystem


class NatureWorld:
    """Persistent deterministic ecology authority and neural teacher."""

    def __init__(self, *, seed: int = 0x4E4154555245, size: int = 64, max_population: int = 180, motion_policy: object|None = None, behavior_policy: object|None = None) -> None:
        if not 24 <= size <= 512 or not 20 <= max_population <= 10_000:
            raise ValueError("nature world bounds drifted")
        self.seed, self.size, self.max_population = int(seed), int(size), int(max_population)
        self.rng = np.random.default_rng(seed)
        self.fields = self._make_fields()
        self.organisms: dict[int, OrganismState] = {}
        self.colonies: dict[int, ColonyState] = {}
        self.next_entity_id = 1
        self.next_colony_id = 1
        self.tick_index = 0
        self.time = 0.0
        self.births = self.deaths = self.predation_events = self.mutation_count = 0
        self.events: list[dict[str, object]] = []
        self.motion_policy = motion_policy
        self.behavior_policy = behavior_policy
        self.materials = MaterialGrid(size,size,seed=seed^0x504F57444552)
        self.colony_ecology = ColonyEcology()
        self.climate = ClimateSystem(seed^0x434C494D415445)

    def _make_fields(self) -> np.ndarray:
        y, x = np.mgrid[:self.size, :self.size]
        phase = (self.seed & 0xFFFF) / 65535 * math.tau
        fields = np.zeros((len(RESOURCE_NAMES), self.size, self.size), dtype=np.float64)
        fields[0] = .46 + .24*np.sin(x*.11+phase)*np.cos(y*.09-phase)
        fields[1] = .55 + .28*np.sin((x+y)*.045+phase)
        fields[2] = .42 + .25*np.cos(x*.07-y*.12+phase)
        fields[3] = .10 + .10*np.maximum(0,np.sin(x*.17+y*.13))
        fields[4] = .08 + .12*np.maximum(0,np.cos(x*.19-y*.16+phase))
        fields[5] = .72 + .10*np.sin(y*.05)
        fields[6] = .22 + .16*np.maximum(0,np.cos(x*.05+phase))
        fields[7] = .02
        fields[8] = .24*np.clip(fields[0]*fields[1],0,1)
        fields[9] = .06
        return np.clip(fields,0,1)

    def seed_founders(self, *, variants_per_family: int = 3, copies: int = 1) -> None:
        genomes = founder_genomes(variants_per_family=variants_per_family)
        for copy in range(copies):
            for ordinal, genome in enumerate(genomes):
                angle = math.tau * (ordinal + copy*.37) / len(genomes)
                radius = self.size * (.20 + .10*copy)
                position = (self.size*.5+math.cos(angle)*radius, self.size*.5+math.sin(angle)*radius)
                self.add_organism(genome, position, energy=.72)

    def add_organism(self, genome: EcoGenome, position: tuple[float,float], *, energy: float=.55, parents: tuple[int,...]=()) -> int:
        if len(self.organisms) >= self.max_population:
            raise RuntimeError("nature population capacity reached")
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        wrapped = (float(position[0])%self.size, float(position[1])%self.size)
        self.organisms[entity_id] = OrganismState.spawn(entity_id,genome,wrapped,birth_tick=self.tick_index,parent_ids=parents,energy=energy)
        return entity_id

    def _cell(self, position: np.ndarray) -> tuple[int,int]:
        return int(position[1])%self.size, int(position[0])%self.size

    def _delta(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        delta = b-a
        delta = (delta+self.size*.5)%self.size-self.size*.5
        return delta

    def _neighbors(self, entity: OrganismState, radius: float) -> list[OrganismState]:
        return [other for other in self.organisms.values() if other.entity_id != entity.entity_id and other.alive and np.linalg.norm(self._delta(entity.position,other.position)) <= radius]

    def _local_gradient(self, resource: int, position: np.ndarray) -> np.ndarray:
        y,x=self._cell(position)
        field=self.fields[resource]
        gx=field[y,(x+1)%self.size]-field[y,(x-1)%self.size]
        gy=field[(y+1)%self.size,x]-field[(y-1)%self.size,x]
        return np.asarray((gx,gy),dtype=np.float64)

    def _choose_intent(self, entity: OrganismState) -> np.ndarray:
        if self.behavior_policy is not None and hasattr(self.behavior_policy,"choose") and not entity.body.incapacitated and entity.energy>=.12:
            neural=self.behavior_policy.choose(entity)
            if neural is not None:
                intent,direction=neural;direction=np.asarray(direction,dtype=np.float64)
                if intent not in ("rest","forage","hunt","flee","mate","follow","photosynthesize","mine","phase_feed","repair","guard","explore") or direction.shape!=(2,) or not np.isfinite(direction).all():raise FloatingPointError("neural nature behavior became invalid")
                entity.intent=intent;return np.clip(direction,-1,1)
        nearby=self._neighbors(entity, 4+entity.genome.trait("perception")*10)
        hunger=max(0.0,.74-entity.energy)
        family=entity.family
        if entity.body.incapacitated or entity.energy < .12:
            entity.intent="repair" if entity.genome.trait("repair")>.35 else "rest"
            return np.zeros(2)
        hostiles=[o for o in nearby if o.family in (0,1,3,4) and o.family != family and o.energy > entity.energy*1.2]
        if hostiles and family in (1,2):
            entity.intent="flee"
            closest=min(hostiles,key=lambda o:np.linalg.norm(self._delta(entity.position,o.position)))
            return -self._delta(entity.position,closest.position)
        if hunger>.04:
            resource=int(np.argmax(np.asarray(entity.genome.diet)*self.fields[:,self._cell(entity.position)[0],self._cell(entity.position)[1]]))
            entity.intent=("photosynthesize" if family==2 else "phase_feed" if family==3 else "mine" if family==4 else "forage")
            direction=self._local_gradient(resource,entity.position)
            prey=[o for o in nearby if self._can_harvest(entity,o)]
            if prey and entity.genome.trait("aggression")>.45:
                entity.intent="hunt"
                return self._delta(entity.position,min(prey,key=lambda o:np.linalg.norm(self._delta(entity.position,o.position))).position)
            return direction
        if entity.stage=="mature" and entity.reproduction_cooldown<=0 and entity.energy>.72:
            mates=[o for o in nearby if o.family==family and o.stage=="mature" and o.reproduction_cooldown<=0 and o.energy>.62]
            if mates:
                entity.intent="mate"
                return self._delta(entity.position,mates[0].position)
        if entity.colony_id in self.colonies:
            center=self.colonies[entity.colony_id].center
            if np.linalg.norm(self._delta(entity.position,center))>5:
                entity.intent="follow"
                return self._delta(entity.position,center)
        entity.intent="explore"
        angle=(entity.genome.developmental.seed*.0001+self.time*.17+entity.entity_id*.71)%math.tau
        return np.asarray((math.cos(angle),math.sin(angle)))

    @staticmethod
    def _can_harvest(predator:OrganismState,prey:OrganismState)->bool:
        if not prey.alive or predator.family==prey.family:return False
        if predator.family==0:return prey.family in (1,2,4) or (prey.family==3 and predator.genome.trait("perception")>.72)
        if predator.family==1:return prey.family==2 or (prey.family in (0,1) and prey.energy<predator.energy*.72)
        return False

    def _move(self, entity: OrganismState, direction: np.ndarray, delta: float) -> None:
        norm=float(np.linalg.norm(direction))
        if norm>1e-9: direction=direction/norm
        locomotion=entity.body.systems()["locomotion"]
        family_speed=(1.05,1.35,.18,1.08,.92)[entity.family]
        stage_scale=.45 if entity.stage=="embryo" else .72 if entity.stage=="juvenile" else .65 if entity.stage=="senescent" else 1.0
        target=direction*family_speed*stage_scale*locomotion
        if self.motion_policy is not None and hasattr(self.motion_policy,"step"):
            neural_target=np.asarray(self.motion_policy.step(entity,target,delta,self.time),dtype=np.float64)
            if neural_target.shape!=(2,) or not np.isfinite(neural_target).all():raise FloatingPointError("neural nature locomotion became invalid")
            target=np.clip(neural_target,-3.2,3.2)
        responsiveness=.22+.58*entity.genome.developmental.traits[6]
        entity.velocity += (target-entity.velocity)*min(1,delta*responsiveness*4)
        entity.velocity *= math.exp(-delta*(.55 if entity.family==3 else 1.15))
        previous=entity.position.copy();proposed=(entity.position+entity.velocity*delta)%self.size
        def solid(position):
            y,x=self._cell(position);return STATE[int(self.materials.material[y,x])]=="solid" and self.materials.structure_id[y,x]>0
        if solid(proposed):
            horizontal=np.asarray((proposed[0],previous[1]));vertical=np.asarray((previous[0],proposed[1]))
            if not solid(horizontal):proposed=horizontal;entity.velocity[1]*=-.12
            elif not solid(vertical):proposed=vertical;entity.velocity[0]*=-.12
            else:proposed=previous;entity.velocity*=-.10
        entity.position=proposed
        if np.linalg.norm(entity.velocity)>.02: entity.heading=math.atan2(entity.velocity[1],entity.velocity[0])
        cost=float(np.linalg.norm(entity.velocity))*entity.genome.trait("move_cost")*.0015*delta
        entity.energy=max(0,entity.energy-cost)

    def _consume(self, entity: OrganismState, delta: float) -> None:
        y,x=self._cell(entity.position)
        diet=np.asarray(entity.genome.diet)
        available=self.fields[:,y,x]
        uptake=np.minimum(available,diet*(.004+.012*entity.genome.developmental.traits[10])*delta)
        if entity.family==2: uptake[8:]=0
        if entity.family==3: uptake[[0,1,5,8,9]]=0
        if entity.family==4: uptake[[0,1,4,5,8,9]]=0
        self.fields[:,y,x]-=uptake
        entity.consumed+=uptake
        quality=float(np.dot(uptake,diet)/max(diet.sum(),1e-8))
        entity.energy=min(1.2,entity.energy+quality*2.4)
        entity.reserve=min(1.0,entity.reserve+quality*.85)
        if entity.family==2:
            self.fields[8,y,x]=min(1.0,self.fields[8,y,x]+delta*.0018*(self.fields[0,y,x]+self.fields[1,y,x]))
            self.fields[5,y,x]=min(1.0,self.fields[5,y,x]+delta*.0007)
        elif entity.family==3:
            self.fields[4,y,x]=min(1.0,self.fields[4,y,x]+delta*.0011)
            self.fields[2,y,x]=min(1.0,self.fields[2,y,x]+delta*.00025)

    def _interactions(self, entity: OrganismState, delta: float) -> None:
        nearby=self._neighbors(entity,1.15)
        if entity.intent=="hunt" and nearby:
            prey=next((o for o in sorted(nearby,key=lambda o:o.entity_id) if self._can_harvest(entity,o)),None)
            if prey is not None:
                damage=(.04+.09*entity.genome.trait("aggression"))*delta
                prey.body.impact((0,0),2.5,min(.22,damage))
                stolen=min(prey.energy,damage*.65)
                prey.energy-=stolen
                entity.energy=min(1.2,entity.energy+stolen*.72)
                self.predation_events+=1
        elif entity.family==3 and entity.energy<.56 and nearby:
            # Anomalies metabolize phase gradients and can nonlocally drain a
            # little organized energy without behaving like ordinary mouths.
            target=min(nearby,key=lambda o:np.linalg.norm(self._delta(entity.position,o.position)))
            drained=min(target.energy,.012*delta*(.5+entity.genome.developmental.traits[14]))
            target.energy-=drained;entity.energy=min(1.2,entity.energy+drained*.82)
            y,x=self._cell(entity.position);self.fields[4,y,x]=min(1,self.fields[4,y,x]+drained*.3)
        if entity.intent=="mate" and entity.gestation_remaining<=0:
            mate=next((o for o in sorted(nearby,key=lambda o:o.entity_id) if o.family==entity.family and o.stage=="mature" and o.reproduction_cooldown<=0 and o.energy>.62),None)
            if mate is not None:
                entity.gestation_remaining=4+entity.genome.trait("gestation")*16
                entity.mate_id=mate.entity_id
                entity.reproduction_cooldown=18+24*(1-entity.genome.trait("fertility"))
                mate.reproduction_cooldown=entity.reproduction_cooldown*.75
                entity.energy-=.12+.16*entity.genome.trait("offspring_investment")
                mate.energy-=.05

    def _resolve_collisions(self) -> None:
        living=sorted((o for o in self.organisms.values() if o.alive),key=lambda o:o.entity_id)
        for left_index,left in enumerate(living):
            for right in living[left_index+1:]:
                if left.family==right.family:continue  # kin can overlap to mate/cluster
                delta=self._delta(left.position,right.position);distance=float(np.linalg.norm(delta));minimum=.54+.22*(left.genome.developmental.traits[0]+right.genome.developmental.traits[0])
                if not 1e-6<distance<minimum:continue
                normal=delta/distance;push=(minimum-distance)*.5
                left.position=(left.position-normal*push)%self.size;right.position=(right.position+normal*push)%self.size
                relative=right.velocity-left.velocity
                impulse=float(np.dot(relative,normal))
                if impulse<0:
                    left.velocity+=normal*impulse*.28;right.velocity-=normal*impulse*.28
                    closing=-impulse
                    if closing>1.05:
                        left_resistance=.35+.45*(left.genome.developmental.traits[3]+left.genome.developmental.traits[5])*.5;right_resistance=.35+.45*(right.genome.developmental.traits[3]+right.genome.developmental.traits[5])*.5;base=min(.075,(closing-1.05)*.026);left_damage=base*(1-left_resistance*.55);right_damage=base*(1-right_resistance*.55);left_cells=left.body.organism.cell_xy;right_cells=right.body.organism.cell_xy;left_point=left_cells[int(np.argmax(left_cells@normal))];right_point=right_cells[int(np.argmin(right_cells@normal))]
                        left.body.impact(tuple(left_point),2.4,left_damage);right.body.impact(tuple(right_point),2.4,right_damage);self.events.append({"tick":self.tick_index,"type":"collision_trauma","left":left.entity_id,"right":right.entity_id,"closing_speed":round(closing,6),"left_damage":round(left_damage,6),"right_damage":round(right_damage,6)})

    def _birth(self, parent: OrganismState) -> None:
        mate=self.organisms.get(parent.mate_id or -1)
        if mate is None: mate=parent
        if len(self.organisms)>=self.max_population: return
        seed=int(self.rng.integers(0,2**63-1))
        try:
            child_genome=recombine(parent.genome,mate.genome,seed=seed,allow_graft=True)
        except (ValueError,OverflowError):
            child_genome=recombine(parent.genome,parent.genome,seed=seed,allow_graft=False)
        offset=self.rng.normal(0,1.2,2)
        child_id=self.add_organism(child_genome,tuple(parent.position+offset),energy=.28+.18*parent.genome.trait("offspring_investment"),parents=(parent.entity_id,mate.entity_id))
        child=self.organisms[child_id]
        child.colony_id=parent.colony_id
        if child.colony_id in self.colonies: self.colonies[child.colony_id].member_ids.add(child_id)
        parent.reserve=max(0,parent.reserve-.18)
        parent.mate_id=None
        parent.gestation_remaining=0
        self.births+=1
        self.mutation_count+=len(child_genome.mutation_log)
        self.events.append({"tick":self.tick_index,"type":"birth","entity":child_id,"parents":child.parent_ids,"mutations":child_genome.mutation_log})

    def _spawn_polyps(self,entity:OrganismState)->int:
        spawned=0
        while entity.polyp_cursor<len(entity.body.polyps):
            record=entity.body.polyps[entity.polyp_cursor];entity.polyp_cursor+=1
            viability=float(record.get("viability",0))
            if viability<.16 or len(self.organisms)>=self.max_population:continue
            seed=int(self.rng.integers(0,2**63-1))
            try:genome=recombine(entity.genome,entity.genome,seed=seed,allow_graft=False)
            except ValueError:genome=entity.genome
            centroid=np.asarray(record.get("centroid",(0,0)),float);direction=centroid/max(float(np.linalg.norm(centroid)),1)
            child_id=self.add_organism(genome,tuple(entity.position+direction*(1.2+viability)),energy=.12+.24*viability,parents=(entity.entity_id,));child=self.organisms[child_id];child.colony_id=entity.colony_id
            if child.colony_id in self.colonies:self.colonies[child.colony_id].member_ids.add(child_id)
            self.births+=1;spawned+=1;self.events.append({"tick":self.tick_index,"type":"polyp","entity":child_id,"parent":entity.entity_id,"viability":round(viability,6),"cells":int(record.get("cell_count",0))})
        return spawned

    def _vegetative_spread(self,entity:OrganismState)->bool:
        if entity.family!=2 or entity.stage!="mature" or entity.energy<.78 or entity.reserve<.35 or entity.reproduction_cooldown>0 or len(self.organisms)>=self.max_population:return False
        nearby=[o for o in self._neighbors(entity,3.2) if o.family==2]
        if len(nearby)>=3:return False
        seed=int(self.rng.integers(0,2**63-1))
        try:genome=recombine(entity.genome,entity.genome,seed=seed,allow_graft=False)
        except ValueError:return False
        # Golden-angle indexed hex directions yield expanding, non-overlapping
        # tessellations without a global plant-grid script.
        direction_index=(entity.entity_id+entity.genome.developmental.generation+self.births)%6;angle=math.tau*direction_index/6;offset=np.asarray((math.cos(angle),math.sin(angle)))*2.35
        child_id=self.add_organism(genome,tuple(entity.position+offset),energy=.24,parents=(entity.entity_id,));child=self.organisms[child_id];child.colony_id=entity.colony_id
        if child.colony_id in self.colonies:self.colonies[child.colony_id].member_ids.add(child_id)
        entity.energy-=.18;entity.reserve-=.22;entity.reproduction_cooldown=32+28*(1-entity.genome.trait("fertility"));self.births+=1;self.mutation_count+=len(genome.mutation_log);self.events.append({"tick":self.tick_index,"type":"vegetative_spread","entity":child_id,"parent":entity.entity_id,"direction":direction_index,"mutations":genome.mutation_log});return True

    def _death_and_decay(self, entity: OrganismState, delta: float) -> None:
        if entity.alive:
            snap=entity.body.tick(min(delta,.5))
            self._spawn_polyps(entity)
            for puddle in entity.body.external_puddle:
                exported=min(float(puddle["amount"]),.018*delta)
                if exported<=0:continue
                puddle["amount"]=float(puddle["amount"])-exported
                material={"vascular":"blood","digestive":"biomass","neural":"blood","phase":"crystal","machine":"oil","root":"sap"}.get(str(puddle["tissue"]),"water")
                self.materials.deposit(material,tuple(entity.position),exported,max(.5,float(puddle["radius"])*.25))
            entity.body.energy=min(entity.body.energy,entity.energy)
            longevity=90+entity.genome.trait("longevity")*310
            if snap.dead or entity.energy<=.002 or entity.age>longevity*1.18:
                entity.alive=False
                entity.stage="dead"
                entity.velocity*=.18
                self.deaths+=1
                self.events.append({"tick":self.tick_index,"type":"death","entity":entity.entity_id,"lineage":entity.genome.lineage_id})
        else:
            entity.decomposition=min(1,entity.decomposition+delta*(.004+.012*(1-entity.genome.trait("cohesion"))))
            y,x=self._cell(entity.position)
            transfer=min(.004*delta,1-entity.decomposition)
            self.fields[9,y,x]=min(1,self.fields[9,y,x]+transfer*entity.body.organism.cell_count*.02)
            self.materials.deposit("biomass",tuple(entity.position),transfer*entity.body.organism.cell_count*.012,.8+entity.decomposition*1.8)
            entity.update_stage()

    @staticmethod
    def _segment_distance(point:np.ndarray,start:np.ndarray,end:np.ndarray)->float:
        delta=end-start;t=float(np.clip(np.dot(point-start,delta)/max(float(np.dot(delta,delta)),1e-8),0,1));return float(np.linalg.norm(point-(start+delta*t)))

    def fire_projectile(self,owner_id:int,target:tuple[float,float],*,speed:float=18,energy:float=1.2)->int:
        owner=self.organisms[owner_id];delta=self._delta(owner.position,np.asarray(target,float));norm=max(float(np.linalg.norm(delta)),1e-8);return self.materials.fire_projectile(tuple(owner.position),tuple(delta/norm*speed),energy=energy,owner_id=owner_id)

    def fire_beam(self,owner_id:int,target:tuple[float,float],*,energy:float=4,width:float=.7)->dict[str,int|float]:
        owner=self.organisms[owner_id];start=owner.position.copy();delta=self._delta(start,np.asarray(target,float));end=start+delta;material_result=self.materials.beam(tuple(start),tuple(end),energy=energy,width=width);bodies=0
        for entity in self.organisms.values():
            if entity.entity_id==owner_id or not entity.alive:continue
            if self._segment_distance(entity.position,start,end)>1.1+width:continue
            local_direction=delta/max(float(np.linalg.norm(delta)),1e-8);normal=np.asarray((-local_direction[1],local_direction[0]));entity.body.cut(tuple(-normal*24),tuple(normal*24),width=min(2.2,.55+width));bodies+=1
        return {**material_result,"bodies_hit":bodies}

    @staticmethod
    def _install_graft(entity:OrganismState,genome:EcoGenome,*,old_appendages:int,old_components:int)->int:
        """Rebuild cellular authority while preserving every surviving old cell."""
        previous=entity.body;replacement=LivingBody(develop(genome.developmental),seed=genome.developmental.seed)
        old_at={tuple(map(int,xy)):index for index,xy in enumerate(previous.organism.cell_xy)}
        installed=0
        for index,xy in enumerate(replacement.organism.cell_xy):
            prior=old_at.get(tuple(map(int,xy)))
            if prior is not None:
                replacement.health[index]=previous.health[prior]
                replacement.scar[index]=previous.scar[prior]
                replacement.fluid[index]=min(replacement.fluid_capacity[index],previous.fluid[prior])
                continue
            component=int(replacement.component_owner[index]);appendage=int(replacement.organism.appendage_index[index])
            if component>=old_components or appendage>=old_appendages:
                replacement.health[index]=.72;replacement.scar[index]=.34;replacement.fluid[index]*=.78;installed+=1
        replacement.energy=min(previous.energy,entity.energy);replacement.external_puddle.extend(previous.external_puddle)
        entity.genome=genome;entity.body=replacement
        return installed

    def graft_from(self,recipient_id:int,donor_id:int,*,kind:str)->dict[str,object]:
        """Physically harvest and install one organ or reciprocal locomotor pair."""
        if recipient_id==donor_id:raise ValueError("graft donor and recipient must differ")
        recipient=self.organisms[recipient_id];donor=self.organisms[donor_id]
        if not recipient.alive or not donor.alive:raise ValueError("grafting requires living bodies")
        seed=int(self.rng.integers(0,2**63-1));old_appendages=len(recipient.genome.developmental.appendages);old_components=len(recipient.genome.developmental.components)
        if kind=="locomotor":
            candidates=[a for a in donor.genome.developmental.appendages if a.paired_with is not None]
            if not candidates:raise ValueError("donor has no reciprocal appendage pair")
            source=sorted(candidates,key=lambda a:(a.kind,a.appendage_id))[0]
            genome=graft_appendage_pair(recipient.genome,donor.genome,source.appendage_id,seed=seed)
            root=np.asarray(next(c.anchor for c in donor.genome.developmental.components if c.component_id==source.root_component),dtype=float)+np.asarray(source.root_offset,dtype=float)
            donor.body.impact(tuple(root),3.2,.66);label=source.kind
        elif kind=="organ":
            candidates=[c for c in donor.genome.developmental.components if c.organ!="none"]
            if not candidates:raise ValueError("donor has no transplantable organ")
            source=sorted(candidates,key=lambda c:(c.organ,c.component_id))[0]
            genome=graft_organ(recipient.genome,donor.genome,source.component_id,seed=seed)
            donor.body.impact(source.anchor,max(1.2,float(max(source.radius))*.9),.58);label=source.organ
        else:raise ValueError("unknown graft kind")
        installed=self._install_graft(recipient,genome,old_appendages=old_appendages,old_components=old_components)
        recipient.energy=max(0,recipient.energy-.08);donor.energy=max(0,donor.energy-.06)
        event={"tick":self.tick_index,"type":"graft","recipient":recipient_id,"donor":donor_id,"kind":kind,"label":label,"installed_cells":installed,"genome":genome.semantic_sha256()}
        self.events.append(event);return event

    def _step_projectiles(self,delta:float)->None:
        previous={p.projectile_id:np.asarray(p.position,float) for p in self.materials.projectiles};self.materials.step(delta)
        for projectile in self.materials.projectiles:
            start=previous.get(projectile.projectile_id,np.asarray(projectile.position,float));end=np.asarray(projectile.position,float)
            for entity in self.organisms.values():
                if not entity.alive or entity.entity_id==projectile.owner_id:continue
                if self._segment_distance(entity.position,start,end)>1+projectile.radius:continue
                relative=self._delta(entity.position,end);local=tuple(relative*5);entity.body.impact(local,2+projectile.radius*2,min(1,projectile.energy*.45));projectile.alive=False;break
        self.materials.projectiles=[p for p in self.materials.projectiles if p.alive]

    def _update_colonies(self) -> None:
        alive=[o for o in self.organisms.values() if o.alive]
        for entity in alive:
            if entity.colony_id is not None: continue
            candidates=[o for o in alive if o.entity_id!=entity.entity_id and o.family==entity.family and o.colony_id is not None and np.linalg.norm(self._delta(entity.position,o.position))<4]
            if candidates and entity.genome.trait("colony_affinity")>.45:
                entity.colony_id=candidates[0].colony_id
                self.colonies[entity.colony_id].member_ids.add(entity.entity_id)
            elif entity.genome.trait("colony_affinity")>.70:
                neighbors=[o for o in alive if o.entity_id!=entity.entity_id and o.family==entity.family and o.colony_id is None and np.linalg.norm(self._delta(entity.position,o.position))<3]
                if neighbors:
                    cid=self.next_colony_id; self.next_colony_id+=1
                    members={entity.entity_id,neighbors[0].entity_id}
                    center=np.mean([self.organisms[i].position for i in members],axis=0)
                    self.colonies[cid]=ColonyState(cid,entity.family,entity.genome.lineage_id,members,center)
                    for i in members:self.organisms[i].colony_id=cid
        for cid in list(self.colonies):
            colony=self.colonies[cid]
            colony.member_ids={i for i in colony.member_ids if i in self.organisms and self.organisms[i].alive}
            if len(colony.member_ids)<2:
                for i in colony.member_ids:self.organisms[i].colony_id=None
                del self.colonies[cid];continue
            colony.center=np.mean([self.organisms[i].position for i in sorted(colony.member_ids)],axis=0)
            if len(colony.member_ids)>14:
                ordered=sorted(colony.member_ids,key=lambda i:(self.organisms[i].position[0],i))
                moved=set(ordered[len(ordered)//2:])
                new_id=self.next_colony_id;self.next_colony_id+=1
                new_center=np.mean([self.organisms[i].position for i in moved],axis=0)
                self.colonies[new_id]=ColonyState(new_id,colony.family,colony.founder_lineage,moved,new_center,colony.generation+1)
                colony.member_ids-=moved;colony.fissions+=1
                for i in moved:self.organisms[i].colony_id=new_id

    def _environment(self, delta: float) -> None:
        # Conservative local diffusion/regeneration; no directional gravity in top-down space.
        for index in (0,2,3,4,5,6,7,8,9):
            field=self.fields[index]
            lap=(np.roll(field,1,0)+np.roll(field,-1,0)+np.roll(field,1,1)+np.roll(field,-1,1)-4*field)
            self.fields[index]=np.clip(field+lap*delta*.008,0,1)
        self.fields[1]=np.clip(self.fields[1]+delta*.00004,0,1)
        self.fields[0]=np.clip(self.fields[0]+delta*.000025,0,1)
        self.fields[2]=np.clip(self.fields[2]+delta*.00001,0,1)

    def step(self, delta: float=.25, *, publish: bool=True) -> WorldSnapshot|None:
        if not math.isfinite(delta) or not .01<=delta<=.5: raise ValueError("nature timestep drifted")
        self.tick_index+=1;self.time+=delta
        self._environment(delta)
        self.climate.step(self,delta)
        if self.behavior_policy is not None and hasattr(self.behavior_policy,"prepare"):self.behavior_policy.prepare(self)
        for entity_id in sorted(list(self.organisms)):
            entity=self.organisms.get(entity_id)
            if entity is None:continue
            entity.age+=delta
            entity.reproduction_cooldown=max(0,entity.reproduction_cooldown-delta)
            entity.update_stage()
            if entity.alive:
                direction=self._choose_intent(entity)
                self._move(entity,direction,delta)
                self._consume(entity,delta)
                self._interactions(entity,delta)
                if (self.tick_index+entity.entity_id*31)%180==0:self._vegetative_spread(entity)
                basal=(.00045+.0018*entity.genome.trait("basal_metabolism"))*delta
                entity.energy=max(0,entity.energy-basal)
                if entity.gestation_remaining>0:
                    entity.gestation_remaining=max(0,entity.gestation_remaining-delta)
                    if entity.gestation_remaining==0:self._birth(entity)
            self._death_and_decay(entity,delta)
            if not entity.finite():raise FloatingPointError(f"entity {entity_id} became non-finite")
        self._update_colonies()
        self.colony_ecology.step(self,delta)
        self._resolve_collisions()
        self._step_projectiles(delta)
        # Decomposed records can leave the active representation after their ledger is complete.
        for entity_id in [i for i,o in self.organisms.items() if o.stage=="decomposed"]:
            del self.organisms[entity_id]
        if len(self.events)>4096:self.events=self.events[-4096:]
        return self.snapshot() if publish else None

    def snapshot(self) -> WorldSnapshot:
        living=[o for o in self.organisms.values() if o.alive]
        family_counts=tuple(sum(o.family==f for o in living) for f in range(len(FAMILIES)))
        lineages=len({o.genome.lineage_id for o in living})
        resource_totals=tuple(round(float(v),6) for v in self.fields.sum(axis=(1,2)))
        records=[]
        for o in sorted(self.organisms.values(),key=lambda x:x.entity_id):
            records.append((o.entity_id,o.genome.semantic_sha256(),tuple(np.round(o.position,6)),tuple(np.round(o.velocity,6)),round(o.age,6),round(o.energy,6),round(o.reserve,6),o.stage,o.intent,o.colony_id,o.alive,round(o.decomposition,6),o.body.snapshot().semantic_sha256))
        payload={"tick":self.tick_index,"time":round(self.time,6),"fields":hashlib.sha256(self.fields.astype("<f8").tobytes()).hexdigest(),"materials":self.materials.semantic_sha256(),"organisms":records,"colonies":[(c.colony_id,c.family,sorted(c.member_ids),tuple(np.round(c.center,6)),c.generation,c.fissions) for c in sorted(self.colonies.values(),key=lambda c:c.colony_id)],"stats":(self.births,self.deaths,self.predation_events,self.mutation_count)}
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=list).encode()).hexdigest()
        return WorldSnapshot(self.tick_index,round(self.time,6),len(living),self.births,self.deaths,self.predation_events,len(self.colonies),lineages,family_counts,resource_totals,self.mutation_count,digest)
