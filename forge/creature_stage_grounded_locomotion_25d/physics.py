from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Callable

import numpy as np

from ..creature_stage_developmental import DevelopedOrganism, pose
from ..creature_stage_developmental.dynamics import _actuator_forces
from ..creature_stage_grounded_locomotion.physics import locomotor_modes, primary_mode
from .contract import Grounded25DConfig


@dataclass(frozen=True,slots=True)
class Grounded25DFrame:
    phase:float
    ground_position:np.ndarray
    ground_velocity:np.ndarray
    heading:np.ndarray
    nodes_local:np.ndarray
    contact_active:np.ndarray
    contact_anchors:np.ndarray
    contact_force:np.ndarray
    muscle_activation:np.ndarray


@dataclass(frozen=True,slots=True)
class Grounded25DRollout:
    organism_sha256:str
    primary_mode:str
    frames:tuple[Grounded25DFrame,...]
    path_length:float
    lateral_extent:float
    depth_extent:float
    maximum_contact_slip:float
    maximum_edge_strain:float
    maximum_chassis_tilt_degrees:float
    appendage_motion_px:float
    contact_switches:int
    identity_sha256:str


def _terminals(organism:DevelopedOrganism)->np.ndarray:
    result=np.full(len(organism.genome.appendages),-1,dtype=np.int16)
    for index in range(len(result)):
        edge_ids=np.flatnonzero(organism.skeleton_edge_appendage==index)
        if edge_ids.size<2:raise ValueError("2.5D appendage lacks articulated terminal")
        result[index]=organism.skeleton_edges[int(edge_ids[-1]),1]
    return result


def _contact_schedule(organism:DevelopedOrganism,modes:tuple[str,...],phase:float)->np.ndarray:
    result=np.zeros(len(modes),dtype=np.bool_)
    for index,mode in enumerate(modes):
        local=(phase+organism.genome.appendages[index].phase)%1
        if mode=="step":result[index]=local<.60
        elif mode=="drag":result[index]=local<.84
        elif mode=="wheel":result[index]=local<.72
    return result


def _control_default(frame:int,total:int)->np.ndarray:
    # Four direction changes prove that depth and lateral traction share one solver.
    angle=math.tau*(frame/max(total-1,1))
    return np.asarray((math.cos(angle),math.sin(angle)),dtype=np.float32)


def _edge_strain(organism:DevelopedOrganism,nodes:np.ndarray,rest:np.ndarray)->float:
    current=np.linalg.norm(nodes[organism.skeleton_edges[:,1]]-nodes[organism.skeleton_edges[:,0]],axis=1)
    # Coincident organ anchors are intentional (for example an anomaly's
    # singularity inside its core).  For those welds, absolute displacement is
    # the meaningful strain measure; dividing by an epsilon invents a huge
    # percentage from a subpixel solver residual.
    return float(np.max(np.abs(current-rest)/np.maximum(rest,1.0)))


def _chassis_tilt(organism:DevelopedOrganism,nodes:np.ndarray)->float:
    count=len(organism.genome.components)
    top=int(np.argmin(organism.skeleton_nodes[:count,1]));bottom=int(np.argmax(organism.skeleton_nodes[:count,1]))
    delta=nodes[top]-nodes[bottom]
    return abs(math.degrees(math.atan2(float(delta[0]),-float(delta[1])))) if abs(float(delta[1]))>1e-7 else 90.0


def simulate_25d(
    organism:DevelopedOrganism,
    config:Grounded25DConfig|None=None,
    *,
    control:Callable[[int,int],np.ndarray]|None=None,
    contact_gate:Callable[[int,float,DevelopedOrganism,tuple[str,...]],np.ndarray]|None=None,
    muscle_gate:Callable[[int,float,DevelopedOrganism],np.ndarray]|None=None,
)->Grounded25DRollout:
    config=config or Grounded25DConfig();control=control or _control_default
    modes=locomotor_modes(organism);mode=primary_mode(organism,modes);terminals=_terminals(organism)
    rest_nodes=organism.skeleton_nodes[:,:2].astype(np.float32)
    nodes=rest_nodes.copy();velocity=np.zeros_like(nodes)
    edges=organism.skeleton_edges
    rest_lengths=np.linalg.norm(rest_nodes[edges[:,1]]-rest_nodes[edges[:,0]],axis=1).astype(np.float32)
    ground_y=max([float(a.endpoint[1]) for a,m in zip(organism.genome.appendages,modes) if m in {"step","drag","wheel"}],default=float(nodes[:,1].max()+3))
    ground=np.zeros(2,dtype=np.float32);ground_velocity=np.zeros(2,dtype=np.float32);heading=np.asarray((0,-1),dtype=np.float32)
    anchors=np.full((len(modes),2),np.nan,dtype=np.float32);previous=np.zeros(len(modes),dtype=np.bool_)
    max_slip=max_strain=max_tilt=path=0.0;switches=0;frames:list[Grounded25DFrame]=[]
    terminal_history=[]
    component_count=len(organism.genome.components)
    for frame_index in range(config.frames):
        phase=(frame_index%60)/60
        desired=np.asarray(control(frame_index,config.frames),dtype=np.float32)
        magnitude=min(1,float(np.linalg.norm(desired)))
        if magnitude>1e-6:
            desired/=magnitude;heading=desired.copy()
        right=np.asarray((-heading[1],heading[0]),dtype=np.float32)
        active=(contact_gate(frame_index,phase,organism,modes) if contact_gate else _contact_schedule(organism,modes,phase)).astype(np.bool_)
        allowed=np.asarray([m in {"step","drag","wheel"} for m in modes],dtype=np.bool_);active&=allowed
        switches+=int(np.count_nonzero(active!=previous))
        for appendage_index in range(len(modes)):
            gene=organism.genome.appendages[appendage_index];local=(phase+gene.phase)%1
            lateral=right*(gene.endpoint[0]*config.ground_scale)
            if active[appendage_index] and not previous[appendage_index]:
                ahead=(.35-local)*config.stride_length
                anchors[appendage_index]=ground+lateral+heading*ahead
            elif not active[appendage_index]:anchors[appendage_index]=np.nan
        traction=[];force=np.zeros((len(modes),2),dtype=np.float32)
        for appendage_index in np.flatnonzero(active):
            gene=organism.genome.appendages[int(appendage_index)];local=(phase+gene.phase)%1
            stance_fraction={"step":.60,"drag":.84,"wheel":.72}[modes[int(appendage_index)]]
            stance_u=local/stance_fraction
            expected=ground+right*(gene.endpoint[0]*config.ground_scale)+heading*((.35-stance_u)*config.stride_length)
            error=anchors[appendage_index]-expected
            gain={"step":19.0,"drag":8.0,"wheel":15.0}[modes[int(appendage_index)]]
            force[appendage_index]=error*gain
            traction.append(force[appendage_index])
        target_velocity=desired*config.maximum_speed*magnitude
        if mode=="float":
            acceleration=(target_velocity-ground_velocity)*config.acceleration*.58
        elif traction:
            acceleration=np.mean(traction,axis=0)+(target_velocity-ground_velocity)*config.acceleration*.16
        else:
            acceleration=-ground_velocity*3.5
        old_ground=ground.copy()
        ground_velocity=(ground_velocity+acceleration*config.dt)*config.body_damping
        speed=float(np.linalg.norm(ground_velocity))
        if speed>config.maximum_speed:ground_velocity*=config.maximum_speed/speed
        ground+=ground_velocity*config.dt;path+=float(np.linalg.norm(ground-old_ground))
        authored=pose(organism,phase)
        muscles=(muscle_gate(frame_index,phase,organism) if muscle_gate else authored.muscle_activation).astype(np.float32)
        target=authored.nodes[:,:2].astype(np.float32)
        previous_nodes=nodes.copy()
        for _ in range(config.substeps):
            drive=np.full((len(nodes),1),.09,dtype=np.float32);drive[:component_count]=.15
            actuator=_actuator_forces(organism,nodes,muscles)*.48
            actuator_norm=np.linalg.norm(actuator,axis=1,keepdims=True)
            actuator*=np.minimum(1.0,.55/np.maximum(actuator_norm,1e-6))
            acceleration_nodes=(target-nodes)*drive+actuator
            if mode!="float":acceleration_nodes[:,1]+=.025
            velocity=velocity*config.node_damping+acceleration_nodes/config.substeps
            velocity_norm=np.linalg.norm(velocity,axis=1,keepdims=True)
            velocity*=np.minimum(1.0,1.2/np.maximum(velocity_norm,1e-6))
            nodes+=velocity/config.substeps
            for _ in range(config.constraint_iterations):
                for edge_index,(left_raw,right_raw) in enumerate(edges):
                    left,right_index=int(left_raw),int(right_raw);delta_node=nodes[right_index]-nodes[left];distance=max(float(np.linalg.norm(delta_node)),1e-6)
                    correction=delta_node*((distance-float(rest_lengths[edge_index]))/distance)*.5
                    nodes[left]+=correction;nodes[right_index]-=correction
                # Upright chassis lock: position only; never face/roll the sprite with heading.
                nodes[0]+=(target[0]-nodes[0])*.62
                for appendage_index in np.flatnonzero(active):
                    terminal=int(terminals[appendage_index]);relative=anchors[appendage_index]-ground
                    visual=np.asarray((float(np.dot(relative,right))/config.ground_scale,ground_y+float(np.dot(relative,heading))*.34),dtype=np.float32)
                    nodes[terminal]=nodes[terminal]*config.contact_compliance+visual*(1-config.contact_compliance)
        velocity=velocity*.35+(nodes-previous_nodes)*.65
        max_strain=max(max_strain,_edge_strain(organism,nodes,rest_lengths));max_tilt=max(max_tilt,_chassis_tilt(organism,nodes))
        for appendage_index in np.flatnonzero(active):
            gene=organism.genome.appendages[int(appendage_index)];terminal=int(terminals[appendage_index]);relative=anchors[appendage_index]-ground
            reconstructed=ground+right*(nodes[terminal,0]*config.ground_scale)+heading*((nodes[terminal,1]-ground_y)/.34)
            max_slip=max(max_slip,float(np.linalg.norm(reconstructed-anchors[appendage_index])))
        terminal_history.append(nodes[terminals].copy())
        frames.append(Grounded25DFrame(phase,ground.copy(),ground_velocity.copy(),heading.copy(),nodes.copy(),active.copy(),anchors.copy(),force.copy(),muscles.copy()))
        previous=active
    history=np.stack(terminal_history)
    appendage_motion=float(np.mean(np.ptp(history,axis=0))) if history.size else 0.0
    positions=np.stack([f.ground_position for f in frames])
    digest=hashlib.sha256(b"nullvector-grounded-25d-rollout-v1\0"+organism.identity_sha256.encode())
    for frame in frames:
        for array in (frame.ground_position,frame.ground_velocity,frame.heading,frame.nodes_local,frame.contact_active,frame.contact_anchors,frame.contact_force,frame.muscle_activation):digest.update(np.ascontiguousarray(array).tobytes())
    return Grounded25DRollout(organism.identity_sha256,mode,tuple(frames),path,float(np.ptp(positions[:,0])),float(np.ptp(positions[:,1])),max_slip,max_strain,max_tilt,appendage_motion,switches,digest.hexdigest())
