from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from .recorder import TeacherTrajectoryRecorder,validate_trajectory
from ..config import PROJECT_ROOT
from ..nature_sim_v2.abilities import entity_abilities,use_ability
from ..nature_sim_v2.demo import NatureDemo
from ..nature_sim_v2.forecast_interventions import apply_intervention
from ..nature_sim_v2.social_actions import bond_nearby

def generate(*,root:Path,session_id:str,steps:int=360,seed:int=0x54454143484552,device:str="cuda"):
    if not 60<=steps<=900:raise ValueError("teacher curriculum steps drifted")
    demo=NatureDemo(seed=seed,device=device,showcase=True);demo.trajectory=TeacherTrajectoryRecorder(root,stride=1,max_frames=steps+8);demo.trajectory.start(session_id,world_seed=demo.world.seed,tick=demo.world.tick_index);actions={}
    for step in range(steps):
        phase=step/38;demo.manual=np.asarray((math.cos(phase),math.sin(phase*.73)))*.82;entity=demo.world.organisms.get(demo.selected)
        living=[item for item in demo.world.organisms.values() if item.alive and item.entity_id!=demo.selected]
        target=min(living,key=lambda item:float(np.linalg.norm(demo.world._delta(entity.position,item.position)))) if entity is not None and living else None
        if entity is not None and target is not None and step%47==9:
            abilities=entity_abilities(entity,equipment_damage=demo.adventure.bonus("damage"));index=(step//47)%len(abilities);demo.action_latch=("ability_up","ability_right","ability_down","ability_left")[min(index,3)];use_ability(demo.world,entity,abilities[index],tuple(target.position),power=.7);actions[demo.action_latch]=actions.get(demo.action_latch,0)+1
        elif entity is not None and target is not None and step%61==17:
            demo.action_latch="projectile";demo.world.fire_projectile(entity.entity_id,tuple(target.position),speed=15,energy=1.15);actions["projectile"]=actions.get("projectile",0)+1
        elif entity is not None and target is not None and step%79==31:
            demo.action_latch="scrape";target.body.impact((0,0),2.1,.18);actions["scrape"]=actions.get("scrape",0)+1
        elif entity is not None and step%83==43:
            demo.action_latch="heal";entity.body.heal((0,0),8,.18);actions["heal"]=actions.get("heal",0)+1
        elif entity is not None and step%101==55:
            demo.action_latch="bond";bond_nearby(demo.world,entity);actions["bond"]=actions.get("bond",0)+1
        elif entity is not None and step%127==73:
            offer=demo.intervention_offers[0]
            if all(demo.adventure.inventory.get(name,0)>=amount for name,amount in offer.costs):
                demo.action_latch="intervention";apply_intervention(demo.world,demo.adventure,entity,offer,forecast_event=demo.timeline_forecast.event);demo.timeline_forecast=demo.timeline_runtime.observe(demo.world,demo.society);demo._refresh_interventions();actions["intervention"]=actions.get("intervention",0)+1
        demo.update(1/30);demo.draw()
    destination=demo.trajectory.finish();demo.pg.quit();manifest=validate_trajectory(destination);report={"format":"nullvector-action-teacher-curriculum/1.0.0","session":session_id,"steps":steps,"seed":seed,"actions":actions,"trajectory_manifest_sha256":manifest["manifest_sha256"],"trajectory_arrays_sha256":manifest["arrays_sha256"],"frames":manifest["frames"]};(destination/"curriculum_report.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8");return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,default=PROJECT_ROOT/"outputs/action_teacher_v1");parser.add_argument("--session",required=True);parser.add_argument("--steps",type=int,default=360);parser.add_argument("--seed",type=int,default=0x54454143484552);parser.add_argument("--device",default="cuda");args=parser.parse_args();print(json.dumps(generate(root=args.root,session_id=args.session,steps=args.steps,seed=args.seed,device=args.device),indent=2))
if __name__=="__main__":main()
