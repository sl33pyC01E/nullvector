from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import numpy as np

from ..creature_stage_developmental import FAMILIES
from .contract import ECO_TRAITS, RESOURCE_NAMES, EcoGenome, WorldSnapshot
from .genetics import founder_genomes, recombine
from .state import ColonyState, OrganismState


class NatureWorld:
    """Persistent deterministic ecology authority and neural teacher."""

    def __init__(self, *, seed: int = 0x4E4154555245, size: int = 64, max_population: int = 180, motion_policy: object|None = None) -> None:
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
        nearby=self._neighbors(entity, 4+entity.genome.trait("perception")*10)
        hunger=max(0.0,.58-entity.energy)
        family=entity.family
        if entity.body.incapacitated or entity.energy < .12:
            entity.intent="repair" if entity.genome.trait("repair")>.35 else "rest"
            return np.zeros(2)
        hostiles=[o for o in nearby if o.family != family and o.energy > entity.energy*1.2]
        if hostiles and family in (1,2):
            entity.intent="flee"
            closest=min(hostiles,key=lambda o:np.linalg.norm(self._delta(entity.position,o.position)))
            return -self._delta(entity.position,closest.position)
        if hunger>.08:
            resource=int(np.argmax(np.asarray(entity.genome.diet)*self.fields[:,self._cell(entity.position)[0],self._cell(entity.position)[1]]))
            entity.intent=("photosynthesize" if family==2 else "phase_feed" if family==3 else "mine" if family==4 else "forage")
            direction=self._local_gradient(resource,entity.position)
            prey=[o for o in nearby if o.family!=family and o.alive and family in (0,1) and o.energy < entity.energy*.92]
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
        entity.position=(entity.position+entity.velocity*delta)%self.size
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
            prey=next((o for o in sorted(nearby,key=lambda o:o.entity_id) if o.family!=entity.family and o.alive),None)
            if prey is not None:
                damage=(.04+.09*entity.genome.trait("aggression"))*delta
                prey.body.impact((0,0),2.5,min(.22,damage))
                stolen=min(prey.energy,damage*.65)
                prey.energy-=stolen
                entity.energy=min(1.2,entity.energy+stolen*.72)
                self.predation_events+=1
        if entity.intent=="mate" and entity.gestation_remaining<=0:
            mate=next((o for o in sorted(nearby,key=lambda o:o.entity_id) if o.family==entity.family and o.stage=="mature" and o.reproduction_cooldown<=0 and o.energy>.62),None)
            if mate is not None:
                entity.gestation_remaining=4+entity.genome.trait("gestation")*16
                entity.mate_id=mate.entity_id
                entity.reproduction_cooldown=18+24*(1-entity.genome.trait("fertility"))
                mate.reproduction_cooldown=entity.reproduction_cooldown*.75
                entity.energy-=.12+.16*entity.genome.trait("offspring_investment")
                mate.energy-=.05

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

    def _death_and_decay(self, entity: OrganismState, delta: float) -> None:
        if entity.alive:
            snap=entity.body.tick(min(delta,.5))
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
            entity.update_stage()

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

    def step(self, delta: float=.25) -> WorldSnapshot:
        if not math.isfinite(delta) or not .01<=delta<=.5: raise ValueError("nature timestep drifted")
        self.tick_index+=1;self.time+=delta
        self._environment(delta)
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
                basal=(.00045+.0018*entity.genome.trait("basal_metabolism"))*delta
                entity.energy=max(0,entity.energy-basal)
                if entity.gestation_remaining>0:
                    entity.gestation_remaining=max(0,entity.gestation_remaining-delta)
                    if entity.gestation_remaining==0:self._birth(entity)
            self._death_and_decay(entity,delta)
            if not entity.finite():raise FloatingPointError(f"entity {entity_id} became non-finite")
        self._update_colonies()
        # Decomposed records can leave the active representation after their ledger is complete.
        for entity_id in [i for i,o in self.organisms.items() if o.stage=="decomposed"]:
            del self.organisms[entity_id]
        if len(self.events)>4096:self.events=self.events[-4096:]
        return self.snapshot()

    def snapshot(self) -> WorldSnapshot:
        living=[o for o in self.organisms.values() if o.alive]
        family_counts=tuple(sum(o.family==f for o in living) for f in range(len(FAMILIES)))
        lineages=len({o.genome.lineage_id for o in living})
        resource_totals=tuple(round(float(v),6) for v in self.fields.sum(axis=(1,2)))
        records=[]
        for o in sorted(self.organisms.values(),key=lambda x:x.entity_id):
            records.append((o.entity_id,o.genome.semantic_sha256(),tuple(np.round(o.position,6)),tuple(np.round(o.velocity,6)),round(o.age,6),round(o.energy,6),round(o.reserve,6),o.stage,o.intent,o.colony_id,o.alive,round(o.decomposition,6),o.body.snapshot().semantic_sha256))
        payload={"tick":self.tick_index,"time":round(self.time,6),"fields":hashlib.sha256(self.fields.astype("<f8").tobytes()).hexdigest(),"organisms":records,"colonies":[(c.colony_id,c.family,sorted(c.member_ids),tuple(np.round(c.center,6)),c.generation,c.fissions) for c in sorted(self.colonies.values(),key=lambda c:c.colony_id)],"stats":(self.births,self.deaths,self.predation_events,self.mutation_count)}
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":"),default=list).encode()).hexdigest()
        return WorldSnapshot(self.tick_index,round(self.time,6),len(living),self.births,self.deaths,self.predation_events,len(self.colonies),lineages,family_counts,resource_totals,self.mutation_count,digest)
