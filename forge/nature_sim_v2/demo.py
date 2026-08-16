from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT
from ..creature_stage_developmental import FAMILIES,TISSUES,develop
from ..creature_stage_neural_locomotion_25d import NeuralLocomotionRuntime
from ..nature_behavior_nn import NeuralBehaviorRuntime
from ..nature_colony_nn import NeuralColonyRuntime
from ..nature_society_nn import NeuralSocietyRuntime
from ..qud_quests_v1 import QuestJournal
from ..qud_society_v1 import SocietyLayer
from ..qud_trade_v1 import execute_trade,generate_trade_offers
from ..nature_world_scale_v1 import InfiniteNatureAtlas,RegionKey
from .body_pose import VisibleBodyPhysics
from .adventure import AdventureState
from .phenotype import phenotype_traits
from .evolution import EvolutionLedger
from .state import ColonyState
from .session_save import load_session,save_session
from .region_store import PersistentRegionStore
from .senses import sensory_field,visible_targets
from .abilities import entity_abilities,use_ability
from .directed_evolution import evolution_offers,metamorphose
from .creature_creator import CreatureCreator
from .succession import choose_successor
from .world import NatureWorld


CHECKPOINT=PROJECT_ROOT/"outputs/creature_stage_neural_locomotion_25d/controller_1200_runtime.pt"
BEHAVIOR_CHECKPOINT=PROJECT_ROOT/"game/generated/models/nature_behavior/controller_v3.pt"
COLONY_CHECKPOINT=PROJECT_ROOT/"game/generated/models/nature_colony/coordinator_v1.pt"
SOCIETY_CHECKPOINT=PROJECT_ROOT/"game/generated/models/nature_society/strategist_v1.pt"
ATLAS=PROJECT_ROOT/"game/generated/anatomical_demo/v2/neural_motion_atlas.png"
QUICK_SAVE=PROJECT_ROOT/"saves/nature_campaign.nvs"
TISSUE_COLORS={"skin":"#58cde0","bone":"#eee4ca","muscle":"#ed5a73","vascular":"#ff416b","respiratory":"#5ce8ff","digestive":"#ffbd4a","neural":"#dc72ff","sensor":"#f4ffff","storage":"#ecd05d","phase":"#aa71ff","root":"#8de05c","machine":"#9cadbd","armor":"#c3ccd6","weapon":"#ff6a50"}
FAMILY_COLORS=("#35dcff","#ff5ca9","#91ff42","#b778ff","#ffb236")
MATERIAL_COLORS=((0,0,0,0),(92,73,46,150),(105,113,117,210),(42,126,174,150),(180,20,58,190),(91,176,62,175),(76,66,43,180),(151,166,178,225),(119,73,63,185),(87,232,148,190),(255,107,26,230),(125,140,150,130),(155,92,255,210))


class NatureDemo:
    def __init__(self,*,seed:int,device:str="cuda",showcase:bool=False) -> None:
        import pygame
        self.pg=pygame;pygame.init();self.screen=pygame.display.set_mode((1440,900),pygame.RESIZABLE);pygame.display.set_caption("Nullvector // Neural Nature Stage v2")
        self.clock=pygame.time.Clock();self.font=pygame.font.SysFont("Consolas",16);self.small=pygame.font.SysFont("Consolas",12);self.big=pygame.font.SysFont("Georgia",30,bold=True)
        self.atlas=pygame.image.load(str(ATLAS)).convert_alpha();self.runtime=NeuralLocomotionRuntime.from_checkpoint(CHECKPOINT,device=device);self.behavior=NeuralBehaviorRuntime.from_checkpoint(BEHAVIOR_CHECKPOINT,device=device);self.colony_runtime=NeuralColonyRuntime.from_checkpoint(COLONY_CHECKPOINT,device=device);self.society_runtime=NeuralSocietyRuntime.from_checkpoint(SOCIETY_CHECKPOINT,device=device)
        self.atlas_world=InfiniteNatureAtlas(seed=seed^0x574F524C44);self.region=RegionKey(0,0);region_seed=self.atlas_world.region_seed(self.region);self.world=NatureWorld(seed=region_seed,size=64,max_population=180,motion_policy=self.runtime,behavior_policy=self.behavior);self.atlas_world.terraform(self.world,self.region);self.world.seed_founders(variants_per_family=3)
        self.region_store=PersistentRegionStore(PROJECT_ROOT/"saves/regions",atlas_seed=self.atlas_world.seed)
        self.world.colony_ecology.role_policy=self.colony_runtime;self.society=SocietyLayer(self.world,seed=seed^0x515544,policy=self.society_runtime);self.last_society_tick=0;self.quests=QuestJournal()
        self.adventure=AdventureState(seed=seed^0x414456,size=self.world.size)
        if showcase:
            self.adventure.inventory.update({"biomass":5.0,"rock":4.0,"metal":4.0,"crystal":3.0,"water":4.0,"knowledge":2.0})
            self.adventure.craft_selected();self.adventure.recipe_index=2;self.adventure.craft_selected();self.adventure.recipe_index=1
            founders=[entity for entity in self.world.organisms.values() if entity.family==0][:3]
            if len(founders)>=3:
                center=np.asarray((20.0,22.0));members=set()
                for index,entity in enumerate(founders):entity.position=center+np.asarray((index*.7,-index*.35));entity.colony_id=1;members.add(entity.entity_id)
                self.world.colonies[1]=ColonyState(1,0,founders[0].genome.lineage_id,members,center.copy());self.world.next_colony_id=2;self.society.found_from_colony(1)
                self.society.step_history(1)
                self.quests.accept_nearest(self.society,self.world,founders[0],self.adventure)
        self.evolution=EvolutionLedger();self.evolution.observe(self.world);self.evolution_epoch=0;self.creator=CreatureCreator();self.creator_seed=seed^0x43524541544F52;self.creator_cache={};self.selected=next(iter(self.world.organisms));self.camera=np.asarray((32.0,32.0));self.zoom=10.0;self.paused=False;self.show_cells=True;self.show_organs=True;self.show_senses=True;self.show_atlas=showcase;self.manual=np.zeros(2);self.tool="inspect";self.message="VAE CELLS + NEURAL BODY + ECOLOGY MIND + COLONY COORDINATOR";self.sprite_cache={};self.visible_physics=VisibleBodyPhysics();self.trade_settlement=None;self.trade_offers=()

    def _enter_region(self,dx,dy,player):
        self.atlas_world.record(self.region,self.world);old_world=self.world;self.region_store.save(self.region,old_world,exclude_entity_id=player.entity_id);self.region=RegionKey(self.region.x+dx,self.region.y+dy,self.region.depth);seed=self.atlas_world.region_seed(self.region);new=self.region_store.load(self.region,motion_policy=self.runtime,behavior_policy=self.behavior,colony_policy=self.colony_runtime)
        resumed=new is not None
        if new is None:new=NatureWorld(seed=seed,size=64,max_population=180,motion_policy=self.runtime,behavior_policy=self.behavior);self.atlas_world.terraform(new,self.region);new.seed_founders(variants_per_family=2)
        entry=(1.2 if dx>0 else 62.8 if dx<0 else float(player.position[0]),1.2 if dy>0 else 62.8 if dy<0 else float(player.position[1]));new_id=new.add_organism(player.genome,entry,energy=player.energy,parents=player.parent_ids);carried=new.organisms[new_id];carried.body=player.body;carried.reserve=player.reserve;carried.age=player.age;carried.stage=player.stage;carried.reproduction_cooldown=player.reproduction_cooldown;carried.velocity=player.velocity.copy()
        carried.heading=player.heading;carried.consumed=player.consumed.copy();carried.neural_contacts=player.neural_contacts.copy();carried.neural_muscles=player.neural_muscles.copy();carried.polyp_cursor=player.polyp_cursor
        for entity_id in old_world.organisms:self.runtime.forget(entity_id)
        self.behavior.cache.clear();self.behavior.last_tick=-1;self.visible_physics.states.clear();self.world=new;self.world.colony_ecology.role_policy=self.colony_runtime;self.selected=new_id;self.camera=carried.position.copy();self.society=SocietyLayer(new,seed=seed^0x515544,policy=self.society_runtime);self.last_society_tick=0
        previous=self.adventure;self.adventure=AdventureState(seed=seed^0x414456,size=new.size);self.adventure.inventory.update(previous.inventory);self.adventure.discoveries|=previous.discoveries;self.adventure.score=previous.score;self.adventure.objectives=previous.objectives;self.adventure.artifacts=list(previous.artifacts);self.adventure.equipped=dict(previous.equipped);self.adventure.recipe_index=previous.recipe_index;self.adventure.craft_count=previous.craft_count
        self.message=f"{'RESUMED' if resumed else 'DISCOVERED'} REGION {self.region.x:+},{self.region.y:+} // {self.atlas_world.describe(self.region).biome.upper()} // PERSISTENT ECOLOGY"

    def world_to_screen(self,position):
        width,height=self.screen.get_size();delta=(np.asarray(position)-self.camera+self.world.size*.5)%self.world.size-self.world.size*.5
        return np.asarray((width*.5+delta[0]*self.zoom,height*.5+delta[1]*self.zoom))

    def screen_to_world(self,position):
        width,height=self.screen.get_size();return (self.camera+(np.asarray(position)-np.asarray((width*.5,height*.5)))/self.zoom)%self.world.size

    def _select(self,mouse):
        point=self.screen_to_world(mouse);living=[o for o in self.world.organisms.values() if o.alive]
        if living:self.selected=min(living,key=lambda o:np.linalg.norm(self.world._delta(point,o.position))).entity_id

    def _damage_at(self,mouse,kind):
        if kind in ("beam","projectile"):
            target=tuple(self.screen_to_world(mouse))
            weapon=1+self.adventure.bonus("damage")
            if kind=="beam":result=self.world.fire_beam(self.selected,target,energy=8*weapon,width=.8);self.adventure.abrade("manipulator",.002);self.message=f"BEAM // {result['cells_hit']} MATERIAL CELLS // {result['bodies_hit']} BODIES"
            else:self.world.fire_projectile(self.selected,target,speed=22,energy=2.4*weapon);self.adventure.abrade("manipulator",.001);self.message="PROJECTILE // PHYSICAL CELL COLLISION"
            return
        self._select(mouse);entity=self.world.organisms.get(self.selected)
        if entity is None:return
        screen=self.world_to_screen(entity.position);local=(np.asarray(mouse)-screen)/3.0;point=(float(local[0]),float(local[1]))
        if kind=="damage":entity.body.impact(point,4,.48*(1+self.adventure.bonus("damage")));self.message="IMPACT // LOCAL TISSUE + ORGAN CAPACITY"
        elif kind=="heal":entity.body.heal(point,7,.28*(1+self.adventure.bonus("repair")));self.adventure.abrade("core",.001);self.message="REPAIR // ENERGY-LIMITED + SCARRING"
        elif kind=="scrape":entity.body.impact(point,2.4,.95);self.message="SCRAPE // SURFACE ABLATION"
        elif kind=="cut":entity.body.cut((point[0]-8,point[1]),(point[0]+8,point[1]),width=.75);self.message="CUT // CELL CONNECTIONS SEVERED"

    def _graft_nearest(self,kind):
        recipient=self.world.organisms.get(self.selected)
        if recipient is None:return
        donors=[o for o in self.world.organisms.values() if o.alive and o.entity_id!=recipient.entity_id]
        if not donors:self.message="GRAFT // NO LIVING DONOR";return
        donor=min(donors,key=lambda o:np.linalg.norm(self.world._delta(recipient.position,o.position)))
        try:
            event=self.world.graft_from(recipient.entity_id,donor.entity_id,kind=kind)
            self.sprite_cache.clear();self.message=f"GRAFT {str(event['label']).upper()} // {event['installed_cells']} PHYSICAL CELLS // DONOR {donor.entity_id}"
        except ValueError as exc:self.message=f"GRAFT REJECTED // {exc}"

    def events(self)->bool:
        pg=self.pg
        for event in pg.event.get():
            if event.type==pg.QUIT:return False
            if event.type==pg.MOUSEWHEEL:self.zoom=float(np.clip(self.zoom*(1.12**event.y),5,24))
            if event.type==pg.MOUSEBUTTONDOWN:
                if event.button==1:
                    if self.tool=="inspect":self._select(event.pos)
                    else:self._damage_at(event.pos,self.tool)
                elif event.button==3:self.camera=self.screen_to_world(event.pos)
            if event.type==pg.KEYDOWN:
                if self.creator.active:
                    if event.key in (pg.K_ESCAPE,pg.K_n):self.creator.active=False;self.message="CREATURE CREATOR CLOSED"
                    elif event.key==pg.K_LEFT:self.creator.change(family=-1)
                    elif event.key==pg.K_RIGHT:self.creator.change(family=1)
                    elif event.key==pg.K_UP:self.creator.change(variant=1)
                    elif event.key==pg.K_DOWN:self.creator.change(variant=-1)
                    elif event.key==pg.K_PAGEUP:self.creator.change(donor=1)
                    elif event.key==pg.K_PAGEDOWN:self.creator.change(donor=-1)
                    elif event.key==pg.K_g:self.creator.cycle_graft()
                    elif event.key==pg.K_TAB:self.creator.cycle_offer_epoch()
                    elif event.key in (pg.K_1,pg.K_2,pg.K_3):self.creator.toggle_offer({pg.K_1:0,pg.K_2:1,pg.K_3:2}[event.key])
                    elif event.key in (pg.K_RETURN,pg.K_KP_ENTER):
                        try:
                            seed=(self.creator_seed+self.creator.revision*104729+self.world.tick_index*65537)&0x7FFF_FFFF_FFFF_FFFF;entity_id=self.creator.incarnate(self.world,self.adventure,self.camera,seed=seed);self.selected=entity_id;self.camera=self.world.organisms[entity_id].position.copy();self.creator.active=False;self.creator.revision+=1;self.sprite_cache.clear();self.message=f"INCARNATED // {FAMILIES[self.world.organisms[entity_id].family].upper()} // {self.world.organisms[entity_id].body.organism.cell_count} LIVING CELLS"
                        except ValueError as exc:self.message=f"INCARNATION REJECTED // {exc}"
                    continue
                if self.trade_settlement is not None:
                    if event.key in (pg.K_ESCAPE,pg.K_y):self.trade_settlement=None;self.trade_offers=();self.message="BARTER CLOSED"
                    elif event.key in (pg.K_7,pg.K_8,pg.K_9):
                        index={pg.K_7:0,pg.K_8:1,pg.K_9:2}[event.key]
                        try:self.message=execute_trade(self.trade_offers[index],settlement=self.society.settlements[self.trade_settlement],adventure=self.adventure,journal=self.quests);self.trade_offers=generate_trade_offers(self.society.settlements[self.trade_settlement],reputation=self.quests.reputation.get(self.society.settlements[self.trade_settlement].faction_id,0),epoch=self.world.tick_index//60)
                        except ValueError as exc:self.message=f"BARTER REJECTED // {exc}"
                    continue
                if event.key==pg.K_ESCAPE:return False
                if event.key==pg.K_SPACE:self.paused=not self.paused
                elif event.key==pg.K_n:self.creator.active=True;self.message="CREATURE CREATOR // ASSEMBLE A CELLULAR LINEAGE"
                elif event.key==pg.K_c:self.show_cells=not self.show_cells
                elif event.key==pg.K_o:self.show_organs=not self.show_organs
                elif event.key==pg.K_m:self.show_atlas=not self.show_atlas
                elif event.key==pg.K_l:self.show_senses=not self.show_senses
                elif event.key==pg.K_F5:
                    report=save_session(world=self.world,adventure=self.adventure,society=self.society,quests=self.quests,atlas=self.atlas_world,region=self.region,selected=self.selected,path=QUICK_SAVE);self.message=f"CAMPAIGN SAVED // CELLS + RELICS + SOCIETIES + CONTRACTS // {report['bytes']/1024:.1f} KIB"
                elif event.key==pg.K_F9 and QUICK_SAVE.is_file():
                    restored=load_session(QUICK_SAVE,motion_policy=self.runtime,behavior_policy=self.behavior,colony_policy=self.colony_runtime,society_policy=self.society_runtime);self.world=restored["world"];self.adventure=restored["adventure"];self.society=restored["society"];self.quests=restored["quests"];self.atlas_world=restored["atlas"];self.region=restored["region"];self.selected=restored["selected"];self.behavior.cache.clear();self.behavior.last_tick=-1;self.visible_physics.states.clear();self.camera=self.world.organisms[self.selected].position.copy();self.last_society_tick=self.world.tick_index;self.evolution=EvolutionLedger();self.evolution.observe(self.world);self.evolution_epoch=self.world.organisms[self.selected].genome.developmental.generation;self.message="CAMPAIGN RESTORED // CELLS + RELICS + SOCIETIES + CONTRACTS + ATLAS EXACT"
                    self.region_store=PersistentRegionStore(PROJECT_ROOT/"saves/regions",atlas_seed=self.atlas_world.seed)
                elif event.key==pg.K_i:self.tool="inspect"
                elif event.key==pg.K_j:self.tool="damage"
                elif event.key==pg.K_h:self.tool="heal"
                elif event.key==pg.K_x:self.tool="scrape"
                elif event.key==pg.K_v:self.tool="cut"
                elif event.key==pg.K_b:self.tool="beam"
                elif event.key==pg.K_p:self.tool="projectile"
                elif event.key==pg.K_g:self._graft_nearest("organ")
                elif event.key==pg.K_k:self._graft_nearest("locomotor")
                elif event.key==pg.K_e and self.selected in self.world.organisms:self.message=self.adventure.interact(self.world,self.world.organisms[self.selected])
                elif event.key==pg.K_q and self.selected in self.world.organisms:
                    try:self.message=self.adventure.build(self.world,self.world.organisms[self.selected])
                    except ValueError as exc:self.message=f"BUILD REJECTED // {exc}"
                elif event.key==pg.K_t:self.message=self.adventure.cycle_recipe()
                elif event.key==pg.K_r:
                    try:self.message=self.adventure.craft_selected()
                    except ValueError as exc:self.message=f"CRAFT NEEDS // {exc}"
                elif event.key==pg.K_u and self.selected in self.world.organisms:self.message=self.quests.accept_nearest(self.society,self.world,self.world.organisms[self.selected],self.adventure)
                elif event.key==pg.K_y and self.selected in self.world.organisms:
                    if not self.society.settlements:self.message="BARTER // NO SETTLEMENTS YET"
                    else:
                        entity=self.world.organisms[self.selected];settlement=min(self.society.settlements.values(),key=lambda item:float(np.linalg.norm(self.world._delta(entity.position,np.asarray(item.center)))));distance=float(np.linalg.norm(self.world._delta(entity.position,np.asarray(settlement.center))))
                        if distance>9:self.message=f"BARTER // SETTLEMENT {distance:.1f} CELLS AWAY"
                        else:self.trade_settlement=settlement.settlement_id;self.trade_offers=generate_trade_offers(settlement,reputation=self.quests.reputation.get(settlement.faction_id,0),epoch=self.world.tick_index//60);self.message="FINITE SETTLEMENT BARTER // 7/8/9 TRADE"
                elif event.key in (pg.K_4,pg.K_5,pg.K_6) and self.adventure.pending_encounter is not None and self.selected in self.world.organisms:
                    index={pg.K_4:0,pg.K_5:1,pg.K_6:2}[event.key]
                    self.message=self.adventure.resolve_pending(index,self.world,self.world.organisms[self.selected])
                elif event.key in (pg.K_1,pg.K_2,pg.K_3) and self.selected in self.world.organisms:
                    entity=self.world.organisms[self.selected];index={pg.K_1:0,pg.K_2:1,pg.K_3:2}[event.key];offer=evolution_offers(entity.genome,epoch=self.evolution_epoch)[index]
                    if self.adventure.inventory["knowledge"]+1e-9<offer.cost:self.message=f"METAMORPHOSIS NEEDS {offer.cost:.1f} KNOWLEDGE"
                    else:
                        seed=(self.world.seed^self.world.tick_index*65537^entity.entity_id*7919^self.evolution_epoch)&0x7FFF_FFFF_FFFF_FFFF;self.adventure.inventory["knowledge"]-=offer.cost;metamorphose(entity,offer,seed=seed);self.runtime.forget(entity.entity_id);self.behavior.cache.pop(entity.entity_id,None);self.visible_physics.states.pop(entity.entity_id,None);self.sprite_cache.clear();self.evolution_epoch+=1;self.world.mutation_count+=1;self.world.events.append({"tick":self.world.tick_index,"type":"directed_evolution","entity":entity.entity_id,"offer":offer.offer_id,"generation":entity.genome.developmental.generation});self.message=f"METAMORPHOSIS // {offer.label.upper()} // WOUNDS MAPPED TO NEW CELLS"
                elif event.key in (pg.K_UP,pg.K_RIGHT,pg.K_DOWN,pg.K_LEFT) and self.selected in self.world.organisms:
                    entity=self.world.organisms[self.selected];abilities=entity_abilities(entity,equipment_damage=self.adventure.bonus("damage"));index={pg.K_UP:0,pg.K_RIGHT:1,pg.K_DOWN:2,pg.K_LEFT:3}[event.key]
                    if index<len(abilities):self.message=use_ability(self.world,entity,abilities[index],tuple(self.screen_to_world(pg.mouse.get_pos())),power=1+self.adventure.bonus("damage"))
                elif event.key==pg.K_f and self.selected in self.world.organisms:self.camera=self.world.organisms[self.selected].position.copy()
        keys=pg.key.get_pressed();self.manual=np.asarray((float(keys[pg.K_d])-float(keys[pg.K_a]),float(keys[pg.K_s])-float(keys[pg.K_w])))
        return True

    def update(self,delta):
        if self.paused:return
        entity=self.world.organisms.get(self.selected)
        if entity is not None and np.linalg.norm(self.manual)>0:
            direction=self.manual/np.linalg.norm(self.manual);entity.velocity+=direction*delta*4.2*(1+self.adventure.bonus("locomotion"));entity.intent="player";self.adventure.abrade("core",delta*.00008)
        previous_position=None if entity is None else entity.position.copy();previous_entity=entity;self.world.step(min(.2,delta*2.2))
        entity=self.world.organisms.get(self.selected)
        if previous_entity is not None and (entity is None or not entity.alive):
            successor,message=choose_successor(self.world,previous_entity)
            if successor is not None:self.selected=successor.entity_id;self.camera=successor.position.copy();self.adventure.succession_count+=1;entity=successor;previous_position=None;self.message=message
            else:self.message=message
        if entity is not None and previous_position is not None:
            jump=entity.position-previous_position;dx=1 if jump[0]<-self.world.size*.5 else -1 if jump[0]>self.world.size*.5 else 0;dy=1 if jump[1]<-self.world.size*.5 else -1 if jump[1]>self.world.size*.5 else 0
            if dx or dy:self._enter_region(dx,dy,entity);entity=self.world.organisms.get(self.selected)
        self.adventure.observe(self.world)
        completed=self.quests.observe(self.world,self.adventure)
        if completed:self.message=f"CONTRACT COMPLETE // {completed[-1].description.upper()}"
        self.evolution.observe(self.world)
        if self.world.tick_index-self.last_society_tick>=60:
            self.society.step_history(1);self.last_society_tick=self.world.tick_index
        if entity is not None and np.linalg.norm(self.manual)>0:self.camera+=(entity.position-self.camera)*min(1,delta*4)

    def _field_background(self):
        pg=self.pg;width,height=self.screen.get_size();surface=pg.Surface((64,64));flora=self.world.fields[8];water=self.world.fields[0];phase=self.world.fields[4];mineral=self.world.fields[2]
        rgb=np.stack((12+mineral*30+phase*34,24+flora*70+water*18,29+water*70+phase*58),axis=2).astype(np.uint8)
        pg.surfarray.blit_array(surface,np.transpose(rgb,(1,0,2)));surface=pg.transform.scale(surface,(int(64*self.zoom),int(64*self.zoom)))
        origin=self.world_to_screen((0,0));self.screen.blit(surface,(int(origin[0]),int(origin[1])))
        self.screen.fill((4,11,15),special_flags=pg.BLEND_RGB_ADD)

    def _draw_materials(self):
        pg=self.pg;surface=pg.Surface((64,64),pg.SRCALPHA)
        for y,x in zip(*np.nonzero(self.world.materials.material)):
            index=int(self.world.materials.material[y,x]);color=MATERIAL_COLORS[index];alpha=int(min(255,color[3]*min(1,float(self.world.materials.mass[y,x]))));surface.set_at((int(x),int(y)),(*color[:3],alpha))
        scaled=pg.transform.scale(surface,(int(64*self.zoom),int(64*self.zoom)));origin=self.world_to_screen((0,0));self.screen.blit(scaled,(int(origin[0]),int(origin[1])))
        for projectile in self.world.materials.projectiles:
            point=self.world_to_screen(projectile.position);pg.draw.circle(self.screen,(255,220,112),(int(point[0]),int(point[1])),max(2,int(projectile.radius*self.zoom)))

    def _sprite(self,entity):
        pg=self.pg;phase=int((self.world.time*(3.2+np.linalg.norm(entity.velocity)*2)+entity.entity_id*.7)%16)
        neural=self.atlas.subsurface(pg.Rect(entity.family*96,phase*96,96,96));sprite=pg.Surface((96,96),pg.SRCALPHA);organism=entity.body.organism;points=self.visible_physics.step(self.world,entity,1/60);rest=organism.cell_xy.astype(np.float32);extent=np.maximum(np.ptp(rest,axis=0),1);scale=min(2.15,66/max(extent));center=np.asarray((48,48))-((rest.min(0)+rest.max(0))*.5*scale)
        for index,xy in enumerate(points):
            health=float(entity.body.health[index])
            if health<=.08:continue
            p=xy*scale+center;sample_x=int(np.clip(p[0],0,95));sample_y=int(np.clip(p[1],0,95));neural_color=neural.get_at((sample_x,sample_y));tissue=pg.Color(TISSUE_COLORS.get(TISSUES[int(organism.tissue[index])],"#ffffff"));color=pg.Color(int(neural_color.r*.62+tissue.r*.38),int(neural_color.g*.62+tissue.g*.38),int(neural_color.b*.62+tissue.b*.38),int(110+145*health));pg.draw.circle(sprite,color,(int(p[0]),int(p[1])),max(1,int(scale*.52)))
        return sprite

    def _draw_entity(self,entity):
        pg=self.pg;point=self.world_to_screen(entity.position);scale=.78+entity.position[1]/self.world.size*.18;size=int(96*scale);sprite=pg.transform.smoothscale(self._sprite(entity),(size,size))
        shadow=pg.Rect(int(point[0]-size*.28),int(point[1]+size*.30),int(size*.56),int(size*.13));pg.draw.ellipse(self.screen,(0,0,0,120),shadow)
        self.screen.blit(sprite,(int(point[0]-size/2),int(point[1]-size*.62)))
        color=pg.Color(FAMILY_COLORS[entity.family]);systems=entity.body.snapshot().systems
        if entity.entity_id==self.selected:pg.draw.rect(self.screen,color,pg.Rect(point[0]-size*.55,point[1]-size*.68,size*1.1,size*1.08),2)
        bar=pg.Rect(point[0]-28,point[1]+size*.42,56,4);pg.draw.rect(self.screen,(16,28,32),bar);pg.draw.rect(self.screen,color,pg.Rect(bar.x,bar.y,bar.w*systems["integrity"],bar.h))
        if entity.colony_id is not None:
            role=self.world.colony_ecology.assignment(entity.entity_id) or "kin";self.screen.blit(self.small.render(f"C{entity.colony_id} {role[:3].upper()}",True,color),(point[0]+22,point[1]-size*.58))

    def _draw_sensory_field(self,entity):
        pg=self.pg;field=sensory_field(entity,equipment_bonus=self.adventure.bonus("perception"));center=self.world_to_screen(entity.position);overlay=pg.Surface(self.screen.get_size(),pg.SRCALPHA);color=pg.Color(FAMILY_COLORS[entity.family]);alpha=int(16+28*field.integrity);radius=field.range*self.zoom
        if field.radial:pg.draw.circle(overlay,(*color[:3],alpha),(int(center[0]),int(center[1])),int(radius));pg.draw.circle(overlay,(*color[:3],90),(int(center[0]),int(center[1])),int(radius),1)
        else:
            angles=np.linspace(entity.heading-field.arc_radians*.5,entity.heading+field.arc_radians*.5,22);points=[tuple(center)]+[(center[0]+math.cos(angle)*radius,center[1]+math.sin(angle)*radius) for angle in angles];pg.draw.polygon(overlay,(*color[:3],alpha),points);pg.draw.lines(overlay,(*color[:3],95),False,points[1:],1)
        self.screen.blit(overlay,(0,0))
        for target_id in visible_targets(self.world,entity,field):
            point=self.world_to_screen(self.world.organisms[target_id].position);pg.draw.circle(self.screen,color,(int(point[0]),int(point[1])),10,1)

    def _draw_settlements(self):
        pg=self.pg
        for settlement in self.society.settlements.values():
            faction=self.society.factions[settlement.faction_id];color=pg.Color(FAMILY_COLORS[faction.family])
            for x,y in settlement.roads:
                point=self.world_to_screen((x,y));pg.draw.circle(self.screen,(75,88,78),(int(point[0]),int(point[1])),max(1,int(self.zoom*.16)))
            center=self.world_to_screen(settlement.center);self.screen.blit(self.small.render(faction.name.upper(),True,color),(center[0]-40,center[1]-18));strategy=self.society.strategies.get(settlement.settlement_id)
            if strategy is not None:self.screen.blit(self.small.render(f"NN {strategy.activity.upper()} > {strategy.project.upper()}",True,(190,132,255)),(center[0]-40,center[1]-4))

    def _draw_ecosystem_links(self):
        pg=self.pg;colors={"pollination":(175,255,89),"root_network":(75,180,91),"phase_charge":(185,105,255),"scavenge":(188,121,72)}
        for link in self.world.ecosystem.links:
            left=self.world.organisms.get(link.left);right=self.world.organisms.get(link.right)
            if left is None or right is None:continue
            start=self.world_to_screen(left.position);delta=self.world._delta(left.position,right.position);end=self.world_to_screen(left.position+delta);color=colors[link.kind];pg.draw.line(self.screen,color,(int(start[0]),int(start[1])),(int(end[0]),int(end[1])),max(1,int(link.strength*2)))

    def _draw_adventure(self):
        pg=self.pg
        for site in self.adventure.sites:
            if site.richness<=.02:continue
            point=self.world_to_screen(site.position);color=(133,255,80) if site.discovered else (83,122,132);pg.draw.circle(self.screen,color,(int(point[0]),int(point[1])),max(3,int(self.zoom*.34)),1)
            if site.discovered:self.screen.blit(self.small.render(site.kind.upper(),True,color),(point[0]+5,point[1]-7))
        x,y=20,92;self.screen.blit(self.font.render(f"EXPEDITION SCORE {self.adventure.score:04} // LIVES {self.adventure.succession_count+1}",True,(183,255,86)),(x,y));y+=24
        for objective in self.adventure.objectives:
            color=(134,255,91) if objective.complete else (121,153,162);mark="[X]" if objective.complete else "[ ]";text=f"{mark} {objective.description} {objective.progress:.0f}/{objective.target:.0f}";self.screen.blit(self.small.render(text,True,color),(x,y));y+=17
        inventory="  ".join(f"{name[:3].upper()} {value:.1f}" for name,value in self.adventure.inventory.items());self.screen.blit(self.small.render(inventory,True,(225,186,91)),(x,y+4));y+=24
        recipe=self.adventure.selected_recipe;self.screen.blit(self.small.render(f"RECIPE [T] {recipe.name.upper()}  [R] CRAFT",True,(196,139,255)),(x,y));y+=17
        for artifact in self.adventure.equipped_artifacts():
            effects=" ".join(f"{name[:3].upper()}+{value:.2f}" for name,value in artifact.effects);self.screen.blit(self.small.render(f"{artifact.slot[:4].upper()} {artifact.name[:28].upper()} {effects}",True,(221,185,255)),(x,y));y+=15
        y+=6;self.screen.blit(self.small.render(f"NATURAL SELECTION // DIV {self.evolution.diversity:.2f} // PAIRS {self.world.breeding.pairings} HYB {self.world.breeding.hybrid_pairings}",True,(111,238,188)),(x,y));y+=17
        for clade in self.evolution.dominant(3):
            self.screen.blit(self.small.render(f"{clade.clade_id[-5:].upper()} F{clade.family} N{clade.population:02} FIT {clade.fitness:.2f} G{clade.max_generation}",True,(99,194,158)),(x,y));y+=15
        selected_entity=self.world.organisms.get(self.selected);culture=None if selected_entity is None or selected_entity.colony_id is None else self.world.colony_culture.states.get(selected_entity.colony_id)
        if culture is not None:self.screen.blit(self.small.render("COLONY MEMORY // "+" ".join(f"{name[:2].upper()}{value:.2f}" for name,value in zip(("forage","defense","medicine","construct","disperse","brood"),culture.values)),True,(101,188,255)),(x,y));y+=15
        y+=6;self.screen.blit(self.small.render(f"CONTRACTS [U] // COMPLETE {self.quests.completed}",True,(255,183,87)),(x,y));y+=17
        for quest in self.quests.active(2):self.screen.blit(self.small.render(f"{quest.metric[:5].upper()} {quest.progress:.0f}/{quest.target:.0f} {quest.description[:30].upper()}",True,(213,150,75)),(x,y));y+=15
        if self.society.settlements:
            selected=self.world.organisms.get(self.selected);settlement=min(self.society.settlements.values(),key=lambda item:float(np.linalg.norm(self.world._delta(selected.position,np.asarray(item.center))))) if selected is not None else next(iter(self.society.settlements.values()));faction=self.society.factions[settlement.faction_id];y+=5;self.screen.blit(self.small.render(f"{faction.name.upper()} // POP {settlement.population} BUILD {len(settlement.buildings)} SHORT {settlement.shortages}",True,(110,207,242)),(x,y));y+=15
            economy=" ".join(f"{name[:3].upper()} {settlement.stockpiles.get(name,0):.1f}" for name in ("food","medicine","parts","energy"));self.screen.blit(self.small.render(economy,True,(90,173,204)),(x,y));y+=15
            strategy=self.society.strategies.get(settlement.settlement_id)
            if strategy is not None:self.screen.blit(self.small.render(f"NEURAL STRATEGY // {strategy.activity.upper()} / {strategy.diplomacy.upper()} / {strategy.project.upper()}",True,(178,125,255)),(x,y));y+=15

    def _draw_encounter(self):
        if self.adventure.pending_encounter is None:return
        pg=self.pg;encounter=self.adventure.encounters[self.adventure.pending_encounter];width=760;height=252;panel=pg.Rect((self.screen.get_width()-width)//2,96,width,height);shade=pg.Surface(self.screen.get_size(),pg.SRCALPHA);shade.fill((0,3,7,118));self.screen.blit(shade,(0,0));pg.draw.rect(self.screen,(5,13,20),panel);pg.draw.rect(self.screen,(190,111,255),panel,2)
        self.screen.blit(self.small.render(f"SITE ENCOUNTER // {encounter.kind.upper()} // BODY SYSTEMS ARE THE DICE",True,(105,220,242)),(panel.x+22,panel.y+16));self.screen.blit(self.big.render(encounter.title.upper(),True,(235,214,255)),(panel.x+22,panel.y+39));self.screen.blit(self.font.render(encounter.description,True,(157,181,190)),(panel.x+22,panel.y+82))
        systems=self.world.organisms[self.selected].body.systems() if self.selected in self.world.organisms else {}
        for index,choice in enumerate(encounter.choices):
            x=panel.x+22+index*242;box=pg.Rect(x,panel.y+118,224,102);pg.draw.rect(self.screen,(8,22,28),box);pg.draw.rect(self.screen,(48,105,118),box,1);self.screen.blit(self.font.render(f"[{index+4}] {choice.label.upper()}",True,(225,238,241)),(x+12,box.y+11));capacity={"perception":systems.get("senses",0),"integrity":systems.get("integrity",0),"neural":systems.get("neural",0)}[choice.approach];risk_color=(116,255,116) if choice.risk<.25 else (255,210,89) if choice.risk<.5 else (255,103,105);self.screen.blit(self.small.render(f"TEST {choice.approach.upper()}  {capacity:.2f}",True,(133,203,218)),(x+12,box.y+42));self.screen.blit(self.small.render(f"RISK {choice.risk:.2f}  YIELD x{choice.reward_scale:.2f}",True,risk_color),(x+12,box.y+62));self.screen.blit(self.small.render("REAL INJURY ON FAILURE",True,(166,117,130)),(x+12,box.y+80))

    def _draw_trade(self):
        if self.trade_settlement is None:return
        pg=self.pg;settlement=self.society.settlements[self.trade_settlement];faction=self.society.factions[settlement.faction_id];width=750;height=258;panel=pg.Rect((self.screen.get_width()-width)//2,102,width,height);shade=pg.Surface(self.screen.get_size(),pg.SRCALPHA);shade.fill((0,4,7,132));self.screen.blit(shade,(0,0));pg.draw.rect(self.screen,(5,16,19),panel);pg.draw.rect(self.screen,(86,225,177),panel,2)
        reputation=self.quests.reputation.get(faction.faction_id,0);self.screen.blit(self.small.render(f"SETTLEMENT BARTER // FINITE PHYSICAL STOCK // REPUTATION {reputation:+.2f}",True,(105,236,191)),(panel.x+22,panel.y+16));self.screen.blit(self.big.render(faction.name.upper(),True,(226,246,235)),(panel.x+22,panel.y+39));self.screen.blit(self.small.render("THEIR SURPLUS BECOMES YOUR SUPPLY; YOUR PAYMENT ENTERS THEIR BUILDING ECONOMY.",True,(130,165,157)),(panel.x+22,panel.y+80))
        for index,offer in enumerate(self.trade_offers):
            x=panel.x+20+index*239;box=pg.Rect(x,panel.y+112,222,112);pg.draw.rect(self.screen,(7,25,25),box);pg.draw.rect(self.screen,(47,104,89),box,1);self.screen.blit(self.font.render(f"[{index+7}] BARTER",True,(205,255,223)),(x+12,box.y+11));self.screen.blit(self.small.render(f"GIVE  {offer.give_amount:.2f} {offer.give_material.upper()}",True,(255,174,105)),(x+12,box.y+43));self.screen.blit(self.small.render(f"TAKE  {offer.receive_amount:.2f} {offer.receive_material.upper()}",True,(104,232,255)),(x+12,box.y+64));have=self.adventure.inventory.get(offer.give_material,0);self.screen.blit(self.small.render(f"YOU HOLD {have:.2f} // [Y] CLOSE",True,(130,156,152)),(x+12,box.y+88))

    def _draw_cells(self,entity):
        pg=self.pg;panel=pg.Rect(self.screen.get_width()-380,72,350,610);pg.draw.rect(self.screen,(5,13,18),panel);pg.draw.rect(self.screen,(38,83,92),panel,1)
        center=np.asarray((panel.centerx,panel.y+165));organism=entity.body.organism;health=entity.body.health;visible=self.visible_physics.cells(entity)
        for index,xy in enumerate(visible):
            if health[index]<=.08:continue
            tissue=TISSUES[int(organism.tissue[index])];color=pg.Color(TISSUE_COLORS.get(tissue,"#ffffff"));color.r=int(color.r*(.25+.75*health[index]));color.g=int(color.g*(.25+.75*health[index]));color.b=int(color.b*(.25+.75*health[index]));p=center+xy*3;pg.draw.circle(self.screen,color,(int(p[0]),int(p[1])),2)
        if self.show_organs:
            for component in organism.genome.components:
                if component.organ=="none":continue
                p=center+np.asarray(component.anchor)*3;pg.draw.circle(self.screen,(255,255,255),(int(p[0]),int(p[1])),max(3,int(max(component.radius)*3)),1)
                self.screen.blit(self.small.render(component.organ,True,(190,220,226)),(p[0]+3,p[1]-7))
        y=panel.y+300;systems=entity.body.snapshot().systems
        for name,value in systems.items():
            self.screen.blit(self.small.render(name.upper(),True,(144,174,183)),(panel.x+14,y));pg.draw.rect(self.screen,(15,33,38),(panel.x+108,y+2,210,8));pg.draw.rect(self.screen,pg.Color(FAMILY_COLORS[entity.family]),(panel.x+108,y+2,int(210*value),8));y+=16
        y+=7;self.screen.blit(self.small.render("HERITABLE PHENOTYPE",True,(218,180,255)),(panel.x+14,y));y+=18
        for trait in phenotype_traits(entity.genome)[:6]:
            self.screen.blit(self.small.render(f"{trait.grade:>3}  {trait.label.upper()}",True,(177,150,225)),(panel.x+14,y));y+=15
        y+=6;self.screen.blit(self.small.render("ANATOMICAL ACTIONS",True,(255,189,83)),(panel.x+14,y));y+=17
        arrows=("UP","RIGHT","DOWN","LEFT")
        for arrow,ability in zip(arrows,entity_abilities(entity,equipment_damage=self.adventure.bonus("damage"))):self.screen.blit(self.small.render(f"{arrow:>5}  {ability.label.upper()}  E{ability.energy_cost:.3f}",True,(221,164,77)),(panel.x+14,y));y+=15

    def _draw_atlas(self):
        pg=self.pg;radius=4;cell=38;records=self.atlas_world.window(self.region,radius);panel=pg.Rect(20,390,cell*(radius*2+1)+20,cell*(radius*2+1)+48);pg.draw.rect(self.screen,(3,10,14),panel);pg.draw.rect(self.screen,(64,127,138),panel,1);self.screen.blit(self.small.render("UNBOUNDED REGION ATLAS // M",True,(93,230,242)),(panel.x+10,panel.y+9));colors={"salt_steppe":"#a8b6a0","fungal_garden":"#cf5da8","glass_dunes":"#e6bc5d","flooded_archive":"#478fd6","phase_reef":"#a768ff","iron_wood":"#8c835e","living_cavern":"#5dbb73","machine_grave":"#8b9baa"}
        for record in records:
            gx=record.key.x-self.region.x+radius;gy=record.key.y-self.region.y+radius;rect=pg.Rect(panel.x+10+gx*cell,panel.y+34+gy*cell,cell-3,cell-3);color=pg.Color(colors[record.biome]);shade=.45+.45*(1-record.danger);color.r=int(color.r*shade);color.g=int(color.g*shade);color.b=int(color.b*shade);pg.draw.rect(self.screen,color,rect);border=(230,255,106) if record.key==self.region else (100,210,220) if record.key in self.atlas_world.visited else (25,45,50);pg.draw.rect(self.screen,border,rect,2 if record.key==self.region else 1);self.screen.blit(self.small.render(record.biome[:2].upper(),True,(3,10,14)),(rect.x+8,rect.y+8))

    def _draw_evolution_offers(self,entity):
        pg=self.pg;offers=evolution_offers(entity.genome,epoch=self.evolution_epoch);width=610;panel=pg.Rect((self.screen.get_width()-width)//2,self.screen.get_height()-98,width,64);pg.draw.rect(self.screen,(4,12,17),panel);pg.draw.rect(self.screen,(93,64,119),panel,1);self.screen.blit(self.small.render(f"DIRECTED METAMORPHOSIS // KNOWLEDGE {self.adventure.inventory['knowledge']:.2f} // HERITABLE + CELLULAR",True,(214,158,255)),(panel.x+10,panel.y+7))
        for index,offer in enumerate(offers):
            x=panel.x+10+index*198;color=(228,185,255) if self.adventure.inventory["knowledge"]>=offer.cost else (102,91,111);self.screen.blit(self.small.render(f"[{index+1}] {offer.label.upper()} ({offer.cost:.1f})",True,color),(x,panel.y+25));self.screen.blit(self.small.render(offer.description[:29].upper(),True,(116,139,148)),(x,panel.y+41))

    def _draw_creator(self):
        pg=self.pg;shade=pg.Surface(self.screen.get_size(),pg.SRCALPHA);shade.fill((0,4,8,225));self.screen.blit(shade,(0,0));width,height=920,610;panel=pg.Rect((self.screen.get_width()-width)//2,(self.screen.get_height()-height)//2,width,height);pg.draw.rect(self.screen,(5,14,20),panel);pg.draw.rect(self.screen,(112,61,145),panel,2);self.screen.blit(self.big.render("CELLULAR CREATURE CREATOR",True,(225,183,255)),(panel.x+28,panel.y+22));self.screen.blit(self.small.render("VALIDATED DEVELOPMENTAL CHASSIS // HERITABLE TRAITS // CROSS-FAMILY GRAFTS",True,(120,191,207)),(panel.x+31,panel.y+64))
        seed=(self.creator_seed+self.creator.revision*104729+self.world.tick_index*65537)&0x7FFF_FFFF_FFFF_FFFF
        try:genome=self.creator.genome(seed=seed);organism=develop(genome.developmental)
        except ValueError as exc:self.screen.blit(self.font.render(f"INVALID BLUEPRINT // {exc}",True,(255,90,90)),(panel.x+40,panel.y+120));return
        preview=pg.Rect(panel.x+28,panel.y+96,430,430);pg.draw.rect(self.screen,(2,8,12),preview);pg.draw.rect(self.screen,(37,75,88),preview,1);xy=organism.cell_xy.astype(float);extent=np.maximum(np.ptp(xy,axis=0),1);scale=min(9,330/max(extent));center=np.asarray(preview.center)-((xy.min(0)+xy.max(0))*.5*scale)
        for point,tissue in zip(xy,organism.tissue):
            p=point*scale+center;color=pg.Color(TISSUE_COLORS[TISSUES[int(tissue)]]);pg.draw.circle(self.screen,color,(int(p[0]),int(p[1])),max(2,int(scale*.42)))
        x=panel.x+492;y=panel.y+106;family=FAMILIES[self.creator.family].upper();donor=FAMILIES[self.creator.donor_family].upper();self.screen.blit(self.font.render(f"CHASSIS  < {family} >",True,pg.Color(FAMILY_COLORS[self.creator.family])),(x,y));y+=30;self.screen.blit(self.font.render(f"MORPHOLOGY  ^ VARIANT {self.creator.variant+1}/6 v",True,(175,211,219)),(x,y));y+=30;self.screen.blit(self.font.render(f"GRAFT [G]  {self.creator.graft_kind.upper()} // DONOR {donor}",True,(216,143,255)),(x,y));y+=34;self.screen.blit(self.small.render("PAGE UP/DOWN CHANGES DONOR",True,(103,130,140)),(x,y));y+=34
        self.screen.blit(self.font.render("HERITABLE ADAPTATIONS [TAB REROLL]",True,(176,239,128)),(x,y));y+=28
        for index,offer in enumerate(self.creator.offers):
            chosen=index in self.creator.selected_offers;color=(167,255,101) if chosen else (126,151,159);self.screen.blit(self.small.render(f"[{'X' if chosen else ' '}] {index+1} {offer.label.upper()}",True,color),(x,y));y+=18;self.screen.blit(self.small.render(offer.description[:48].upper(),True,(92,119,128)),(x+18,y));y+=24
        organs=sorted({item.organ for item in genome.developmental.components if item.organ!="none"});appendages={kind:sum(item.kind==kind for item in genome.developmental.appendages) for kind in sorted({item.kind for item in genome.developmental.appendages})};self.screen.blit(self.small.render("ORGANS // "+" ".join(name.upper() for name in organs[:7]),True,(244,184,115)),(x,y));y+=21;self.screen.blit(self.small.render("APPENDAGES // "+" ".join(f"{name.upper()}×{count}" for name,count in appendages.items()),True,(244,184,115)),(x,y));y+=26
        cost=self.creator.cost();cost_text=" + ".join(f"{amount:.2f} {name.upper()}" for name,amount in cost.items());affordable=all(self.adventure.inventory.get(name,0)>=amount for name,amount in cost.items());self.screen.blit(self.font.render("COST // "+cost_text,True,(134,255,94) if affordable else (255,105,95)),(x,y));y+=32;self.screen.blit(self.font.render(f"{organism.cell_count} CELLS // {len(organism.muscles)} MUSCLES // {len(organs)} ORGAN SYSTEMS",True,(90,218,239)),(x,y));self.screen.blit(self.font.render("ENTER // INCARNATE    N / ESC // CANCEL",True,(225,225,232)),(panel.x+250,panel.bottom-46))

    def draw(self):
        pg=self.pg;self.screen.fill((3,9,13));self._field_background()
        self._draw_materials()
        self._draw_ecosystem_links()
        self._draw_settlements()
        self._draw_adventure()
        if self.show_atlas:self._draw_atlas()
        selected_entity=self.world.organisms.get(self.selected)
        if self.show_senses and selected_entity is not None and selected_entity.alive:self._draw_sensory_field(selected_entity)
        for entity in sorted(self.world.organisms.values(),key=lambda o:(o.position[1],o.entity_id)):
            if entity.alive:self._draw_entity(entity)
        width=self.screen.get_width();pg.draw.rect(self.screen,(3,10,14),(0,0,width,58));self.screen.blit(self.big.render("NULLVECTOR // NATURE",True,(229,245,246)),(22,12))
        snap=self.world.snapshot();biome=self.atlas_world.describe(self.region).biome.upper();climate=self.world.climate.current;network=self.world.ecosystem;status=f"REG {self.region.x:+04},{self.region.y:+04} {biome[:10]:10} {climate.season.upper():9} POP {snap.population:03} B{snap.births:03} D{snap.deaths:03} C{snap.colony_count:02} M{snap.mutation_count:02} SYM {network.pollinations}/{network.root_transfers}/{network.phase_couplings}"
        self.screen.blit(self.font.render(status,True,(75,227,255)),(470,20));self.screen.blit(self.small.render("WASD MOVE  ARROWS ANATOMICAL ACTIONS  E INTERACT  Q BUILD  T/R CRAFT  U CONTRACT  M ATLAS  L SENSES  F5/F9 SAVE  J/H/X/V TOOLS",True,(133,164,174)),(20,self.screen.get_height()-24))
        entity=self.world.organisms.get(self.selected)
        if entity is not None:
            if self.show_cells:self._draw_cells(entity)
            self._draw_evolution_offers(entity)
        if self.adventure.pending_encounter is not None:self._draw_encounter()
        if self.trade_settlement is not None:self._draw_trade()
        if self.creator.active:self._draw_creator()
        self.screen.blit(self.small.render(f"TOOL {self.tool.upper()} // {self.message}",True,(255,196,80)),(22,66));pg.display.flip()

    def run(self,*,capture:Path|None=None)->None:
        if capture is not None:
            for _ in range(3):self.update(1/30)
            self.draw();capture.parent.mkdir(parents=True,exist_ok=True);self.pg.image.save(self.screen,str(capture));return
        running=True
        while running:
            delta=min(.05,self.clock.tick(60)/1000);running=self.events();self.update(delta);self.draw()
        self.pg.quit()


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,default=0x51554944);parser.add_argument("--device",default="cuda");parser.add_argument("--capture",type=Path);parser.add_argument("--showcase",action="store_true");parser.add_argument("--creator",action="store_true");parser.add_argument("--encounter",action="store_true");parser.add_argument("--trade",action="store_true");args=parser.parse_args();demo=NatureDemo(seed=args.seed,device=args.device,showcase=args.showcase);demo.creator.active=args.creator
    if args.encounter:
        entity=demo.world.organisms[demo.selected];site=next(item for item in demo.adventure.sites if item.kind=="phase_well");entity.position=site.position.copy();demo.message=demo.adventure.interact(demo.world,entity)
    if args.trade and demo.society.settlements:
        settlement=next(iter(demo.society.settlements.values()));demo.trade_settlement=settlement.settlement_id;demo.trade_offers=generate_trade_offers(settlement,reputation=demo.quests.reputation.get(settlement.faction_id,0),epoch=0);demo.message="FINITE SETTLEMENT BARTER // 7/8/9 TRADE"
    demo.run(capture=args.capture)


if __name__=="__main__":main()
