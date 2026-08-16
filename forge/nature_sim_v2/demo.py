from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT
from ..creature_stage_developmental import FAMILIES,TISSUES
from ..creature_stage_neural_locomotion_25d import NeuralLocomotionRuntime
from ..nature_behavior_nn import NeuralBehaviorRuntime
from ..qud_society_v1 import SocietyLayer
from .body_pose import VisibleBodyPhysics
from .adventure import AdventureState
from .world import NatureWorld


CHECKPOINT=PROJECT_ROOT/"outputs/creature_stage_neural_locomotion_25d/controller_1200_runtime.pt"
BEHAVIOR_CHECKPOINT=PROJECT_ROOT/"game/generated/models/nature_behavior/controller_v3.pt"
ATLAS=PROJECT_ROOT/"game/generated/anatomical_demo/v1/neural_motion_atlas.png"
TISSUE_COLORS={"skin":"#58cde0","bone":"#eee4ca","muscle":"#ed5a73","vascular":"#ff416b","respiratory":"#5ce8ff","digestive":"#ffbd4a","neural":"#dc72ff","sensor":"#f4ffff","storage":"#ecd05d","phase":"#aa71ff","root":"#8de05c","machine":"#9cadbd","armor":"#c3ccd6","weapon":"#ff6a50"}
FAMILY_COLORS=("#35dcff","#ff5ca9","#91ff42","#b778ff","#ffb236")
MATERIAL_COLORS=((0,0,0,0),(92,73,46,150),(105,113,117,210),(42,126,174,150),(180,20,58,190),(91,176,62,175),(76,66,43,180),(151,166,178,225),(119,73,63,185),(87,232,148,190),(255,107,26,230),(125,140,150,130),(155,92,255,210))


class NatureDemo:
    def __init__(self,*,seed:int,device:str="cuda") -> None:
        import pygame
        self.pg=pygame;pygame.init();self.screen=pygame.display.set_mode((1440,900),pygame.RESIZABLE);pygame.display.set_caption("Nullvector // Neural Nature Stage v2")
        self.clock=pygame.time.Clock();self.font=pygame.font.SysFont("Consolas",16);self.small=pygame.font.SysFont("Consolas",12);self.big=pygame.font.SysFont("Georgia",30,bold=True)
        self.atlas=pygame.image.load(str(ATLAS)).convert_alpha();self.runtime=NeuralLocomotionRuntime.from_checkpoint(CHECKPOINT,device=device);self.behavior=NeuralBehaviorRuntime.from_checkpoint(BEHAVIOR_CHECKPOINT,device=device)
        self.world=NatureWorld(seed=seed,size=64,max_population=180,motion_policy=self.runtime,behavior_policy=self.behavior);self.world.seed_founders(variants_per_family=3)
        self.society=SocietyLayer(self.world,seed=seed^0x515544);self.last_society_tick=0
        self.adventure=AdventureState(seed=seed^0x414456,size=self.world.size)
        self.selected=next(iter(self.world.organisms));self.camera=np.asarray((32.0,32.0));self.zoom=10.0;self.paused=False;self.show_cells=True;self.show_organs=True;self.manual=np.zeros(2);self.tool="inspect";self.message="VAE CELLS + 4.46M BODY CONTROL + 3.56M ECOLOGY MIND";self.sprite_cache={};self.visible_physics=VisibleBodyPhysics()

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
            if kind=="beam":result=self.world.fire_beam(self.selected,target,energy=8,width=.8);self.message=f"BEAM // {result['cells_hit']} MATERIAL CELLS // {result['bodies_hit']} BODIES"
            else:self.world.fire_projectile(self.selected,target,speed=22,energy=2.4);self.message="PROJECTILE // PHYSICAL CELL COLLISION"
            return
        self._select(mouse);entity=self.world.organisms.get(self.selected)
        if entity is None:return
        screen=self.world_to_screen(entity.position);local=(np.asarray(mouse)-screen)/3.0;point=(float(local[0]),float(local[1]))
        if kind=="damage":entity.body.impact(point,4,.48);self.message="IMPACT // LOCAL TISSUE + ORGAN CAPACITY"
        elif kind=="heal":entity.body.heal(point,7,.28);self.message="REPAIR // ENERGY-LIMITED + SCARRING"
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
                if event.key==pg.K_ESCAPE:return False
                if event.key==pg.K_SPACE:self.paused=not self.paused
                elif event.key==pg.K_c:self.show_cells=not self.show_cells
                elif event.key==pg.K_o:self.show_organs=not self.show_organs
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
                elif event.key==pg.K_f and self.selected in self.world.organisms:self.camera=self.world.organisms[self.selected].position.copy()
        keys=pg.key.get_pressed();self.manual=np.asarray((float(keys[pg.K_d])-float(keys[pg.K_a]),float(keys[pg.K_s])-float(keys[pg.K_w])))
        return True

    def update(self,delta):
        if self.paused:return
        entity=self.world.organisms.get(self.selected)
        if entity is not None and np.linalg.norm(self.manual)>0:
            direction=self.manual/np.linalg.norm(self.manual);entity.velocity+=direction*delta*4.2;entity.intent="player"
        self.world.step(min(.2,delta*2.2))
        self.adventure.observe(self.world)
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
        if entity.colony_id is not None:self.screen.blit(self.small.render(f"C{entity.colony_id}",True,color),(point[0]+22,point[1]-size*.58))

    def _draw_settlements(self):
        pg=self.pg
        for settlement in self.society.settlements.values():
            faction=self.society.factions[settlement.faction_id];color=pg.Color(FAMILY_COLORS[faction.family])
            for building in settlement.buildings:
                origin=self.world_to_screen(building.origin);rect=pg.Rect(origin[0],origin[1],building.width*self.zoom,building.height*self.zoom);pg.draw.rect(self.screen,(8,18,21),rect);pg.draw.rect(self.screen,color,rect,max(1,int(self.zoom*.16)))
            for x,y in settlement.roads:
                point=self.world_to_screen((x,y));pg.draw.circle(self.screen,(75,88,78),(int(point[0]),int(point[1])),max(1,int(self.zoom*.16)))
            center=self.world_to_screen(settlement.center);self.screen.blit(self.small.render(faction.name.upper(),True,color),(center[0]-40,center[1]-18))

    def _draw_adventure(self):
        pg=self.pg
        for site in self.adventure.sites:
            if site.richness<=.02:continue
            point=self.world_to_screen(site.position);color=(133,255,80) if site.discovered else (83,122,132);pg.draw.circle(self.screen,color,(int(point[0]),int(point[1])),max(3,int(self.zoom*.34)),1)
            if site.discovered:self.screen.blit(self.small.render(site.kind.upper(),True,color),(point[0]+5,point[1]-7))
        x,y=20,92;self.screen.blit(self.font.render(f"EXPEDITION SCORE {self.adventure.score:04}",True,(183,255,86)),(x,y));y+=24
        for objective in self.adventure.objectives:
            color=(134,255,91) if objective.complete else (121,153,162);mark="[X]" if objective.complete else "[ ]";text=f"{mark} {objective.description} {objective.progress:.0f}/{objective.target:.0f}";self.screen.blit(self.small.render(text,True,color),(x,y));y+=17
        inventory="  ".join(f"{name[:3].upper()} {value:.1f}" for name,value in self.adventure.inventory.items());self.screen.blit(self.small.render(inventory,True,(225,186,91)),(x,y+4))

    def _draw_cells(self,entity):
        pg=self.pg;panel=pg.Rect(self.screen.get_width()-380,72,350,390);pg.draw.rect(self.screen,(5,13,18),panel);pg.draw.rect(self.screen,(38,83,92),panel,1)
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

    def draw(self):
        pg=self.pg;self.screen.fill((3,9,13));self._field_background()
        self._draw_materials()
        self._draw_settlements()
        self._draw_adventure()
        for entity in sorted(self.world.organisms.values(),key=lambda o:(o.position[1],o.entity_id)):
            if entity.alive:self._draw_entity(entity)
        width=self.screen.get_width();pg.draw.rect(self.screen,(3,10,14),(0,0,width,58));self.screen.blit(self.big.render("NULLVECTOR // NATURE",True,(229,245,246)),(22,12))
        snap=self.world.snapshot();status=f"POP {snap.population:03}  BIRTH {snap.births:03}  DEATH {snap.deaths:03}  HUNT {snap.predation_events:04}  COL {snap.colony_count:02}  FAC {len(self.society.factions):02}  MUT {snap.mutation_count:02}"
        self.screen.blit(self.font.render(status,True,(75,227,255)),(470,20));self.screen.blit(self.small.render("WASD PLAY  E INTERACT  Q BUILD  F FOLLOW  J DAMAGE  H HEAL  X SCRAPE  V CUT  B BEAM  P PROJECTILE  G ORGAN GRAFT  K LIMB GRAFT  C CELLS  O ORGANS",True,(133,164,174)),(20,self.screen.get_height()-24))
        entity=self.world.organisms.get(self.selected)
        if entity is not None and self.show_cells:self._draw_cells(entity)
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
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,default=0x51554944);parser.add_argument("--device",default="cuda");parser.add_argument("--capture",type=Path);args=parser.parse_args();NatureDemo(seed=args.seed,device=args.device).run(capture=args.capture)


if __name__=="__main__":main()
