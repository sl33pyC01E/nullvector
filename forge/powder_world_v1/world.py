from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import numpy as np

from .contract import CORROSION,DENSITY,FLAMMABILITY,MATERIALS,STATE,VISCOSITY,Projectile


class MaterialGrid:
    """Top-down cellular material authority: liquids form radial puddles."""
    def __init__(self,width:int,height:int,*,seed:int=0)->None:
        if not 16<=width<=2048 or not 16<=height<=2048:raise ValueError("powder world bounds drifted")
        self.width,self.height,self.seed=width,height,int(seed);self.material=np.zeros((height,width),np.uint8);self.mass=np.zeros((height,width),np.float32);self.temperature=np.full((height,width),.18,np.float32);self.damage=np.zeros((height,width),np.float32);self.structure_id=np.zeros((height,width),np.int32);self.projectiles:list[Projectile]=[];self.next_projectile=1;self.tick_index=0

    def deposit(self,material:str,center:tuple[float,float],amount:float,radius:float)->float:
        if material not in MATERIALS or not math.isfinite(amount) or amount<=0 or not .25<=radius<=100:raise ValueError("powder deposit drifted")
        index=MATERIALS.index(material);cy,cx=float(center[1]),float(center[0]);y0=max(0,int(cy-radius));y1=min(self.height,int(cy+radius)+1);x0=max(0,int(cx-radius));x1=min(self.width,int(cx+radius)+1);yy,xx=np.mgrid[y0:y1,x0:x1];weight=np.clip(1-np.hypot(xx-cx,yy-cy)/radius,0,1);weight/=max(float(weight.sum()),1e-8);addition=weight*amount;target=self.mass[y0:y1,x0:x1];empty=(target<=1e-6)|(self.material[y0:y1,x0:x1]==index);accepted=addition*empty;target+=accepted;self.material[y0:y1,x0:x1][accepted>0]=index;return float(accepted.sum())

    def add_structure(self,mask:np.ndarray,*,structure_id:int,material:str="rock")->None:
        if mask.shape!=self.material.shape or mask.dtype!=np.bool_ or structure_id<=0 or STATE[MATERIALS.index(material)]!="solid":raise ValueError("powder structure drifted")
        # Reject isolated single-pixel hazards; structures must be conglomerated.
        neighbors=np.zeros_like(mask,np.uint8)
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):neighbors+=np.roll(np.roll(mask,dy,0),dx,1)
        if np.any(mask&(neighbors<2)):raise ValueError("powder structure contains a shearing singleton")
        index=MATERIALS.index(material);self.material[mask]=index;self.mass[mask]=1;self.structure_id[mask]=structure_id

    def _diffuse_liquids(self,delta:float)->None:
        change=np.zeros_like(self.mass)
        for index,state in enumerate(STATE):
            if state not in ("liquid","gas"):continue
            selected=self.material==index
            if not selected.any():continue
            value=np.where(selected,self.mass,0);rate=(.22 if state=="gas" else .10)*(1-VISCOSITY[index]*.65)*delta
            average=(np.roll(value,1,0)+np.roll(value,-1,0)+np.roll(value,1,1)+np.roll(value,-1,1))/4
            flux=(average-value)*rate;change+=flux
        self.mass=np.maximum(self.mass+change,0)
        vacancies=self.mass<=0;self.material[vacancies]=0
        # Propagate liquid identity into newly wetted empty cells from the
        # strongest cardinal neighbor, with no preferred downward direction.
        wet=(self.mass>0)&(self.material==0)
        for y,x in zip(*np.nonzero(wet)):
            candidates=[(float(self.mass[ny%self.height,nx%self.width]),int(self.material[ny%self.height,nx%self.width])) for ny,nx in ((y-1,x),(y+1,x),(y,x-1),(y,x+1)) if STATE[int(self.material[ny%self.height,nx%self.width])]=="liquid"]
            if candidates:self.material[y,x]=max(candidates)[1]

    def _react(self,delta:float)->None:
        fire=self.material==MATERIALS.index("fire");self.temperature[fire]+=delta*.4
        hot=self.temperature>.68
        for index,flammability in enumerate(FLAMMABILITY):
            ignite=(self.material==index)&hot
            if flammability>0:
                burned=np.minimum(self.mass[ignite],flammability*delta*.05);self.mass[ignite]-=burned;self.temperature[ignite]+=burned*.8
                exhausted=ignite&(self.mass<=.02);self.material[exhausted]=MATERIALS.index("smoke");self.mass[exhausted]=.12
        acid=self.material==MATERIALS.index("acid")
        adjacent=np.zeros_like(acid)
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):adjacent|=np.roll(np.roll(acid,dy,0),dx,1)
        corrodible=adjacent&np.isin(self.material,[MATERIALS.index("metal"),MATERIALS.index("biomass"),MATERIALS.index("rock")]);self.damage[corrodible]+=delta*.025
        destroyed=corrodible&(self.damage>=1);self.material[destroyed]=0;self.mass[destroyed]=0;self.structure_id[destroyed]=0
        self.temperature+=(np.roll(self.temperature,1,0)+np.roll(self.temperature,-1,0)+np.roll(self.temperature,1,1)+np.roll(self.temperature,-1,1)-4*self.temperature)*delta*.03;self.temperature=np.clip(self.temperature,0,2)

    def fire_projectile(self,position:tuple[float,float],velocity:tuple[float,float],*,radius:float=.65,energy:float=1,material:str="metal",owner_id:int|None=None)->int:
        if material not in MATERIALS or not .1<=radius<=8 or not 0<energy<=100:raise ValueError("powder projectile drifted")
        projectile=Projectile(self.next_projectile,position,velocity,radius,energy,MATERIALS.index(material),owner_id);self.next_projectile+=1;self.projectiles.append(projectile);return projectile.projectile_id

    def beam(self,start:tuple[float,float],end:tuple[float,float],*,energy:float,width:float=.75)->dict[str,float|int]:
        delta=np.asarray(end,float)-np.asarray(start,float);length=max(float(np.linalg.norm(delta)),1e-6);steps=max(1,int(length*3));hit=0;spent=0.0
        for t in np.linspace(0,1,steps):
            point=np.asarray(start)*(1-t)+np.asarray(end)*t;x,y=int(point[0])%self.width,int(point[1])%self.height
            radius=max(0,int(math.ceil(width)))
            for yy in range(y-radius,y+radius+1):
                for xx in range(x-radius,x+radius+1):
                    py,px=yy%self.height,xx%self.width
                    if self.material[py,px]==0:continue
                    resistance=.18+.82*DENSITY[int(self.material[py,px])];damage=min(energy-spent,.08/max(resistance,.05));self.damage[py,px]+=damage;spent+=damage*.25;hit+=1
                    if self.damage[py,px]>=1:self.material[py,px]=0;self.mass[py,px]=0;self.structure_id[py,px]=0
                    if spent>=energy:return {"cells_hit":hit,"energy_spent":spent}
        return {"cells_hit":hit,"energy_spent":spent}

    def _step_projectiles(self,delta:float)->None:
        for projectile in self.projectiles:
            if not projectile.alive:continue
            position=np.asarray(projectile.position,float);velocity=np.asarray(projectile.velocity,float);distance=float(np.linalg.norm(velocity))*delta;substeps=max(1,int(distance*2));step=velocity*delta/substeps
            for _ in range(substeps):
                position=(position+step)%np.asarray((self.width,self.height));x,y=int(position[0]),int(position[1])
                if self.material[y,x]!=0:
                    resistance=.2+DENSITY[int(self.material[y,x])];self.damage[y,x]+=projectile.energy/max(resistance,1e-4);projectile.energy*=.18;projectile.alive=False
                    if self.damage[y,x]>=1:self.material[y,x]=0;self.mass[y,x]=0;self.structure_id[y,x]=0
                    break
            projectile.position=(float(position[0]),float(position[1]))
        self.projectiles=[p for p in self.projectiles if p.alive]

    def step(self,delta:float=.1)->None:
        if not .001<=delta<=.5:raise ValueError("powder timestep drifted")
        self.tick_index+=1;self._diffuse_liquids(delta);self._react(delta);self._step_projectiles(delta)
        if not np.isfinite(self.mass).all() or np.any(self.mass<0):raise FloatingPointError("powder material state invalid")

    def semantic_sha256(self)->str:
        digest=hashlib.sha256(b"nullvector-powder-world-v1\0")
        for array in (self.material,self.mass,self.temperature,self.damage,self.structure_id):digest.update(np.ascontiguousarray(array).tobytes())
        for p in self.projectiles:digest.update(repr(p).encode())
        return digest.hexdigest()
