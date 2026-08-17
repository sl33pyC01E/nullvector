from __future__ import annotations

import hashlib,math
import numpy as np
from ..creature_stage_developmental.dynamics import _actuator_forces
from ..creature_stage_developmental.motion import skin_cells
from ..creature_stage_grounded_locomotion.contract import GroundedLocomotionConfig
from ..creature_stage_grounded_locomotion.physics import (GroundedCycle,GroundedFrame,_edge_strain,_inverse_mass,
 _project_edges_and_contacts,_terminal_nodes,_traction,_vertical_axis_degrees,dominant_family,locomotor_modes,primary_mode)

def _target_nodes(organism,terminal_targets):
    target=organism.skeleton_nodes[:,:2].astype(np.float32,copy=True)
    for owner in range(len(organism.genome.appendages)):
        edge_ids=np.flatnonzero(organism.skeleton_edge_appendage==owner)
        chain=[int(organism.skeleton_edges[int(edge_ids[0]),0])]+[int(organism.skeleton_edges[int(e),1]) for e in edge_ids]
        delta=np.asarray(terminal_targets[owner],np.float32)-target[chain[-1]]
        for rank,node in enumerate(chain[1:],1): target[node]+=delta*(rank/(len(chain)-1))
    return target

def simulate_target_field_cycle(organism,policy,config=None):
    config=config or GroundedLocomotionConfig();modes=locomotor_modes(organism);primary=primary_mode(organism,modes);terminals=_terminal_nodes(organism)
    ground=max([float(organism.genome.appendages[i].endpoint[1]) for i,m in enumerate(modes) if m in {"step","drag","wheel"}],default=float(organism.skeleton_nodes[:,1].max()+3))
    pos=organism.skeleton_nodes[:,:2].astype(np.float32,copy=True);vel=np.zeros_like(pos);inv=_inverse_mass(organism)
    rest=np.linalg.norm(organism.skeleton_nodes[organism.skeleton_edges[:,1],:2]-organism.skeleton_nodes[organism.skeleton_edges[:,0],:2],axis=1).astype(np.float32)
    anchors=np.full((len(modes),2),np.nan,np.float32);previous=np.zeros(len(modes),np.bool_);body_x=body_v=0.;recorded=[];slip=strain=vertical=work=0.;seam_ref=None;start=None
    anomaly=float(organism.genome.family_mix[3]);components=len(organism.genome.components)
    for cycle in range(config.settle_cycles+1):
      for fi in range(config.frame_count):
        phase=fi/config.frame_count;local=pos.copy();local[:,0]-=body_x
        muscles,active,term_target=policy.predict(organism,local,vel,previous,phase,body_v)
        active=np.asarray(active,np.bool_)&np.asarray([m in {"step","drag","wheel"} for m in modes]);target_local=_target_nodes(organism,term_target)
        for i in range(len(modes)):
          if active[i] and not previous[i]: anchors[i]=(pos[int(terminals[i]),0],ground)
          elif not active[i]: anchors[i]=np.nan
        reaction=np.zeros((len(modes),2),np.float32)
        for i in np.flatnonzero(active):
          if modes[i]=="wheel":
            gene=organism.genome.appendages[i];u=((phase+gene.phase)%1)/config.wheel_stance_fraction;contact_x=gene.endpoint[0]-4.4*u
          else: contact_x=target_local[int(terminals[i]),0]
          reaction[i,0]=np.clip((anchors[i,0]-contact_x-body_x)*_traction(modes[i],config),-.42,.42)
        drive=float(reaction[active,0].mean()) if bool(active.any()) else 0.;body_v=body_v*config.body_damping+drive+config.float_drive*anomaly*anomaly;body_v=float(np.clip(body_v,-config.maximum_body_speed,config.maximum_body_speed));body_x+=body_v;work+=abs(drive*body_v)
        target=target_local.copy();target[:,0]+=body_x;old=pos.copy()
        for _ in range(config.substeps):
          direct=np.full((len(pos),1),.075,np.float32);direct[:components]=.115;acc=(target-pos)*direct+_actuator_forces(organism,pos,muscles)*.72
          if primary!="float":acc[:,1]+=config.gravity
          vel=vel*config.node_damping+acc/config.substeps;pos+=vel/config.substeps;_project_edges_and_contacts(organism,pos,rest,inv,target[0],terminals,active,anchors,config)
        vel=vel*.3+(pos-old)*.7
        for i in np.flatnonzero(active):slip=max(slip,float(np.linalg.norm(pos[int(terminals[i])]-anchors[i])))
        strain=max(strain,_edge_strain(organism,pos,rest));local=pos.copy();local[:,0]-=body_x;vertical=max(vertical,_vertical_axis_degrees(organism,local))
        if cycle==config.settle_cycles-1 and fi==0:seam_ref=local.copy()
        if cycle==config.settle_cycles:
          if start is None:start=body_x
          cells=skin_cells(organism,pos);cells[:,0]-=body_x;recorded.append(GroundedFrame(phase,pos.copy(),local,cells,vel.copy(),body_x,body_v,active.copy(),anchors.copy(),reaction.copy(),np.asarray(muscles).copy()))
        previous=active
    distance=float(recorded[-1].body_world_x-start);seam=float(np.max(np.abs(recorded[0].nodes_local-seam_ref)));d=hashlib.sha256(b"target-field-cycle-v1")
    for f in recorded:d.update(f.nodes_local.tobytes())
    return GroundedCycle(organism.identity_sha256,modes,primary,tuple(recorded),float(ground),distance,distance/71,seam,strain,slip,work,vertical,d.hexdigest())
