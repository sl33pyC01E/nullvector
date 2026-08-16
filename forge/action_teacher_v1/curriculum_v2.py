from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from .contract import ACTIONS
from .recorder import TeacherTrajectoryRecorder,validate_trajectory
from ..config import PROJECT_ROOT
from ..nature_sim_v2.abilities import entity_abilities,use_ability
from ..nature_sim_v2.demo import NatureDemo
from ..nature_sim_v2.directed_evolution import evolution_offers,metamorphose
from ..nature_sim_v2.forecast_interventions import apply_intervention
from ..nature_sim_v2.social_actions import bond_nearby

def _entities(demo):
    living=sorted((item for item in demo.world.organisms.values() if item.alive),key=lambda item:item.entity_id)
    if not living:raise RuntimeError("teacher world lost every organism")
    entity=demo.world.organisms.get(demo.selected)
    if entity is None or not entity.alive:entity=living[0];demo.selected=entity.entity_id
    targets=[item for item in living if item.entity_id!=entity.entity_id]
    target=min(targets,key=lambda item:float(np.linalg.norm(demo.world._delta(entity.position,item.position)))) if targets else entity
    return entity,target,living

def _ability_actor(demo,living,index):
    current=demo.world.organisms.get(demo.selected)
    candidates=sorted(living,key=lambda item:(item.entity_id!=getattr(current,"entity_id",-1),item.entity_id))
    for candidate in candidates:
        if len(entity_abilities(candidate,equipment_damage=demo.adventure.bonus("damage")))>index:
            return candidate
    return None

def _apply(demo,action,step):
    entity,target,living=_entities(demo);demo.teacher_aim_override=target.position.copy();world=demo.world
    if action in ("none","inspect"):return True
    if action=="impact":target.body.impact((0,0),3,.24)
    elif action=="heal":entity.body.impact((0,0),2,.12);entity.body.heal((0,0),8,.22)
    elif action=="scrape":target.body.impact((0,0),2.1,.48)
    elif action=="cut":target.body.cut((-12,0),(12,0),width=.72)
    elif action=="beam":world.fire_beam(entity.entity_id,tuple(target.position),energy=4.5,width=.65)
    elif action=="projectile":world.fire_projectile(entity.entity_id,tuple(target.position),speed=16,energy=1.4)
    elif action=="interact":
        site=demo.adventure.sites[step%len(demo.adventure.sites)];entity.position=site.position.copy();demo.adventure.interact(world,entity)
    elif action=="build":demo.adventure.inventory.update({"rock":8,"metal":8,"biomass":8});demo.adventure.build(world,entity)
    elif action=="craft":demo.adventure.inventory.update({"rock":8,"metal":8,"biomass":8,"crystal":8});demo.adventure.craft_selected()
    elif action=="bond":
        kin=next((item for item in living if item.entity_id!=entity.entity_id and item.family==entity.family),None)
        if kin is None:return False
        kin.position=(entity.position+np.asarray((.5,.2)))%world.size;bond_nearby(world,entity)
    elif action in ("graft_organ","graft_locomotor"):
        donor=next((item for item in living if item.family!=entity.family),None)
        if donor is None:return False
        world.graft_from(entity.entity_id,donor.entity_id,kind="organ" if action=="graft_organ" else "locomotor");demo.runtime.forget(entity.entity_id);demo.visible_physics.states.pop(entity.entity_id,None)
    elif action.startswith("ability_"):
        index=("ability_up","ability_right","ability_down","ability_left").index(action);entity=_ability_actor(demo,living,index)
        if entity is None:return False
        demo.selected=entity.entity_id;targets=[item for item in living if item.entity_id!=entity.entity_id];target=min(targets,key=lambda item:float(np.linalg.norm(demo.world._delta(entity.position,item.position)))) if targets else entity;demo.teacher_aim_override=target.position.copy();abilities=entity_abilities(entity,equipment_damage=demo.adventure.bonus("damage"))
        use_ability(world,entity,abilities[index],tuple(target.position),power=.8)
    elif action=="intervention":
        demo.adventure.inventory.update({"rock":8,"metal":8,"biomass":8,"crystal":8,"water":8,"knowledge":8});offer=demo.intervention_offers[step%len(demo.intervention_offers)];apply_intervention(world,demo.adventure,entity,offer,forecast_event=demo.timeline_forecast.event)
    elif action in ("trade","service"):
        if not demo.society.settlements:return False
        settlement=next(iter(demo.society.settlements.values()));entity.position=np.asarray(settlement.center,dtype=np.float64)
        if action=="service":entity.body.impact((0,0),2,.12);entity.body.heal((0,0),8,.18)
        else:demo.adventure.inventory["biomass"]+=.2;settlement.stockpiles["biomass"]=settlement.stockpiles.get("biomass",0)+.2
    elif action=="metamorphosis":
        offer=evolution_offers(entity.genome,epoch=step)[step%3];metamorphose(entity,offer,seed=(demo.world.seed^step*7919)&0x7fff_ffff_ffff_ffff);demo.runtime.forget(entity.entity_id);demo.behavior.cache.pop(entity.entity_id,None);demo.visible_physics.states.pop(entity.entity_id,None)
    else:return False
    return True

def generate(*,root:Path,session_id:str,steps:int=132,seed:int=0x414354494F4E5632,device:str="cuda"):
    if not 88<=steps<=440:raise ValueError("balanced teacher curriculum steps drifted")
    demo=NatureDemo(seed=seed,device=device,showcase=True);demo.trajectory=TeacherTrajectoryRecorder(root,stride=1,max_frames=steps+8);demo.trajectory.start(session_id,world_seed=demo.world.seed,tick=demo.world.tick_index);counts={name:0 for name in ACTIONS};attempts={name:0 for name in ACTIONS}
    schedule=tuple(ACTIONS[index%len(ACTIONS)] for index in range(steps))
    for step,requested in enumerate(schedule):
        phase=step/17;demo.manual=np.asarray((math.cos(phase),math.sin(phase*.71)),np.float32)*(.35+.55*((step%9)/8));attempts[requested]+=1
        try:success=_apply(demo,requested,step)
        except (ValueError,RuntimeError,IndexError):success=False
        action=requested if success else "none";demo.action_latch=action;counts[action]+=1;demo.update(1/30);demo.draw()
    destination=demo.trajectory.finish();demo.pg.quit();manifest=validate_trajectory(destination);missing=[name for name in ACTIONS if name not in ("trade","service") and counts[name]<1]
    if missing:raise RuntimeError("balanced teacher missed actions: "+",".join(missing))
    report={"format":"nullvector-action-teacher-balanced-curriculum/2.0.0","session":session_id,"steps":steps,"seed":seed,"actions":counts,"attempts":attempts,"trajectory_manifest_sha256":manifest["manifest_sha256"],"trajectory_arrays_sha256":manifest["arrays_sha256"],"frames":manifest["frames"]};(destination/"curriculum_v2_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8");return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=PROJECT_ROOT/"outputs/action_teacher_v1");parser.add_argument("--session",required=True);parser.add_argument("--steps",type=int,default=132);parser.add_argument("--seed",type=int,default=0x414354494F4E5632);parser.add_argument("--device",default="cuda");args=parser.parse_args();print(json.dumps(generate(root=args.root,session_id=args.session,steps=args.steps,seed=args.seed,device=args.device),indent=2))
if __name__=="__main__":main()
