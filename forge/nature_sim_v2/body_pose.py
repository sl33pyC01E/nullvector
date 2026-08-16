from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from ..creature_stage_grounded_locomotion.physics import locomotor_modes


@dataclass(slots=True)
class VisibleBodyState:
    identity_sha256:str
    nodes:np.ndarray
    velocity:np.ndarray
    anchors:np.ndarray
    previous_contact:np.ndarray
    nearest:np.ndarray
    weights:np.ndarray
    cells:np.ndarray
    rest:np.ndarray
    edge_left:np.ndarray
    edge_right:np.ndarray
    lengths:np.ndarray
    component_count:int
    terminals:np.ndarray
    grounded_modes:np.ndarray
    muscle_origin:np.ndarray
    muscle_insertion:np.ndarray
    muscle_strength:np.ndarray


class VisibleBodyPhysics:
    """Online 2.5D cell/skeleton solver driven by neural muscles and contacts."""
    def __init__(self)->None:self.states:dict[int,VisibleBodyState]={}

    @staticmethod
    def _register(entity)->VisibleBodyState:
        organism=entity.body.organism;rest=organism.skeleton_nodes[:,:2].astype(np.float32);points=organism.cell_xy.astype(np.float32);distance=np.linalg.norm(points[:,None]-rest[None],axis=2);count=min(3,distance.shape[1]);nearest=np.argpartition(distance,kth=count-1,axis=1)[:,:count];selected=np.take_along_axis(distance,nearest,axis=1);weights=np.exp(-selected*.72);weights/=np.maximum(weights.sum(1,keepdims=True),1e-8);left=organism.skeleton_edges[:,0].astype(np.int32);right=organism.skeleton_edges[:,1].astype(np.int32);lengths=np.linalg.norm(rest[right]-rest[left],axis=1).astype(np.float32);terminals=VisibleBodyPhysics._terminals(organism);grounded=np.asarray([mode in {"step","drag","wheel"} for mode in locomotor_modes(organism)],np.bool_);muscles=np.asarray(organism.muscles,np.float32);origin=muscles[:,0].astype(np.int32);insertion=muscles[:,1].astype(np.int32);strength=muscles[:,3].astype(np.float32)
        return VisibleBodyState(organism.identity_sha256,rest.copy(),np.zeros_like(rest),np.full((len(organism.genome.appendages),2),np.nan,np.float32),np.zeros(len(organism.genome.appendages),np.bool_),nearest,weights,points.copy(),rest,left,right,lengths,len(organism.genome.components),terminals,grounded,origin,insertion,strength)

    @staticmethod
    def _terminals(organism)->np.ndarray:
        result=np.full(len(organism.genome.appendages),-1,np.int16)
        for index in range(len(result)):
            edges=np.flatnonzero(organism.skeleton_edge_appendage==index)
            if edges.size:result[index]=organism.skeleton_edges[int(edges[-1]),1]
        return result

    def step(self,world,entity,delta:float)->np.ndarray:
        organism=entity.body.organism;state=self.states.get(entity.entity_id)
        if state is None or state.identity_sha256!=organism.identity_sha256:state=self._register(entity);self.states[entity.entity_id]=state
        rest=state.rest;component_count=state.component_count;terminals=state.terminals
        muscles=np.asarray(entity.neural_muscles,np.float32)
        if muscles.shape!=(len(organism.muscles),):muscles=np.zeros(len(organism.muscles),np.float32)
        contact=np.asarray(entity.neural_contacts,np.bool_)
        if contact.shape!=(len(organism.genome.appendages),):contact=np.zeros(len(organism.genome.appendages),np.bool_)
        contact&=state.grounded_modes
        for index in range(len(contact)):
            if contact[index] and not state.previous_contact[index]:state.anchors[index]=entity.position+np.asarray((organism.genome.appendages[index].endpoint[0]/6,0),np.float32)
            elif not contact[index]:state.anchors[index]=np.nan
        target=rest.copy();target[:component_count,1]+=math.sin(world.time*2.4+entity.entity_id*.31)*.13
        delta_node=state.nodes[state.muscle_insertion]-state.nodes[state.muscle_origin];muscle_length=np.maximum(np.linalg.norm(delta_node,axis=1),1e-6);normal=np.stack((-delta_node[:,1],delta_node[:,0]),axis=1)/muscle_length[:,None];muscle_force=normal*state.muscle_strength[:,None]*muscles[:,None]*.102;force=np.zeros_like(state.nodes);np.add.at(force,state.muscle_insertion,muscle_force);np.add.at(force,state.muscle_origin,-muscle_force*.35);force_norm=np.linalg.norm(force,axis=1,keepdims=True);force*=np.minimum(1,.52/np.maximum(force_norm,1e-6));acceleration=(target-state.nodes)*.055+force
        if entity.family!=3:acceleration[:,1]+=.012
        rate=min(1,delta*60);state.velocity=state.velocity*.72+acceleration*rate;state.nodes+=state.velocity*rate
        for _ in range(6):
            vector=state.nodes[state.edge_right]-state.nodes[state.edge_left];distance=np.maximum(np.linalg.norm(vector,axis=1),1e-6);correction=vector*((distance-state.lengths)/distance)[:,None]*.5;node_correction=np.zeros_like(state.nodes);np.add.at(node_correction,state.edge_left,correction);np.add.at(node_correction,state.edge_right,-correction);state.nodes+=node_correction
            state.nodes[0]=target[0]
            for index in np.flatnonzero(contact):
                terminal=int(terminals[index]);relative=world._delta(entity.position,state.anchors[index]);offset=np.asarray((relative[0]*6,relative[1]*2.4),np.float32);magnitude=float(np.linalg.norm(offset))
                if magnitude>6:offset*=6/magnitude
                state.nodes[terminal]=state.nodes[terminal]*.08+(rest[terminal]+offset)*.92
        state.velocity*=.38;delta_nodes=state.nodes-rest;state.cells=organism.cell_xy.astype(np.float32)+(delta_nodes[state.nearest]*state.weights[:,:,None]).sum(1);state.previous_contact=contact.copy();return state.cells

    def cells(self,entity)->np.ndarray:
        state=self.states.get(entity.entity_id);return entity.body.organism.cell_xy.astype(np.float32) if state is None else state.cells

    def forget(self,entity_id:int)->None:self.states.pop(entity_id,None)
