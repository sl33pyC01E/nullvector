from __future__ import annotations
import hashlib,json,os,shutil,uuid
from pathlib import Path
import numpy as np
from .contract import ACTIONS,COUNTERFACTUAL_SHAPE,FORMAT,FRAME_SIZE,STATE_FEATURES,canonical,source_sha256

DISK_FLOOR=100*1024**3

def _array_digest(arrays):
    digest=hashlib.sha256(b"nullvector-action-teacher-arrays-v1\0")
    for name,array in sorted(arrays.items()):digest.update(name.encode()+b"\0"+str(array.dtype).encode()+b"\0"+str(array.shape).encode()+b"\0"+array.tobytes())
    return digest.hexdigest()

class TeacherTrajectoryRecorder:
    def __init__(self,root:Path,*,stride:int=3,max_frames:int=900):
        if stride<1 or not 1<=max_frames<=10000:raise ValueError("trajectory recorder bounds drifted")
        self.root=Path(root);self.stride=int(stride);self.max_frames=int(max_frames);self.active=False;self.session_id="";self.world_seed=0;self.start_tick=0;self.last_tick=-10**12;self._rows={name:[] for name in ("frame","state","control","action","selected","timeline_event","timeline","counterfactual","tick")}
    @property
    def frame_count(self)->int:return len(self._rows["tick"])
    def start(self,session_id:str,*,world_seed:int,tick:int)->None:
        if self.active or not session_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in session_id):raise ValueError("trajectory session identity drifted")
        if self.root.exists() and shutil.disk_usage(self.root).free<DISK_FLOOR:raise RuntimeError("trajectory recorder reached 100 GiB disk floor")
        for value in self._rows.values():value.clear()
        self.active=True;self.session_id=session_id;self.world_seed=int(world_seed);self.start_tick=int(tick);self.last_tick=-10**12
    def append(self,*,frame,state,control,action:str,selected:int,timeline_event:int,timeline,counterfactual,tick:int)->bool:
        if not self.active:return False
        if action not in ACTIONS:raise ValueError("trajectory action vocabulary drifted")
        if int(tick)-self.last_tick<self.stride and action=="none":return False
        if len(self._rows["tick"])>=self.max_frames:return False
        frame=np.asarray(frame);state=np.asarray(state,np.float32);control=np.asarray(control,np.float32);timeline=np.asarray(timeline,np.float32);counterfactual=np.asarray(counterfactual,np.float32)
        if frame.shape!=(FRAME_SIZE[1],FRAME_SIZE[0],3) or frame.dtype!=np.uint8:raise ValueError("trajectory frame contract drifted")
        if state.shape!=(STATE_FEATURES,) or control.shape!=(4,) or timeline.shape!=(3,) or counterfactual.shape!=COUNTERFACTUAL_SHAPE:raise ValueError("trajectory tensor shape drifted")
        if not all(np.isfinite(value).all() for value in (state,control,timeline,counterfactual)):raise ValueError("trajectory tensor contains nonfinite values")
        values=(frame.copy(),state.copy(),control.copy(),ACTIONS.index(action),int(selected),int(timeline_event),timeline.copy(),counterfactual.copy(),int(tick))
        for name,value in zip(self._rows,values):self._rows[name].append(value)
        self.last_tick=int(tick);return True
    def finish(self)->Path:
        if not self.active or not self._rows["tick"]:raise ValueError("trajectory recorder has no active frames")
        self.root.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(self.root).free<DISK_FLOOR:raise RuntimeError("trajectory recorder reached 100 GiB disk floor")
        destination=self.root/self.session_id
        if destination.exists():raise FileExistsError(destination)
        staging=self.root/f".{self.session_id}.tmp-{uuid.uuid4().hex}";staging.mkdir()
        arrays={"frame":np.stack(self._rows["frame"]).astype(np.uint8),"state":np.stack(self._rows["state"]).astype(np.float32),"control":np.stack(self._rows["control"]).astype(np.float32),"action":np.asarray(self._rows["action"],np.uint8),"selected":np.asarray(self._rows["selected"],np.int64),"timeline_event":np.asarray(self._rows["timeline_event"],np.uint8),"timeline":np.stack(self._rows["timeline"]).astype(np.float32),"counterfactual":np.stack(self._rows["counterfactual"]).astype(np.float32),"tick":np.asarray(self._rows["tick"],np.int64)}
        semantic=_array_digest(arrays);archive=staging/"trajectory.npz";np.savez_compressed(archive,**arrays);artifact=hashlib.sha256(archive.read_bytes()).hexdigest();manifest={"format":FORMAT,"source_sha256":source_sha256(),"session_id":self.session_id,"world_seed":self.world_seed,"start_tick":self.start_tick,"end_tick":int(arrays["tick"][-1]),"frames":len(arrays["tick"]),"stride":self.stride,"frame_size":list(FRAME_SIZE),"actions":list(ACTIONS),"arrays_sha256":semantic,"artifact":{"path":"trajectory.npz","bytes":archive.stat().st_size,"sha256":artifact},"shapes":{name:list(array.shape) for name,array in arrays.items()},"dtypes":{name:str(array.dtype) for name,array in arrays.items()}}
        manifest["manifest_sha256"]=hashlib.sha256(canonical(manifest)).hexdigest();(staging/"manifest.json").write_bytes(canonical(manifest));os.replace(staging,destination);self.active=False;return destination

def validate_trajectory(path:Path)->dict:
    root=Path(path);manifest=json.loads((root/"manifest.json").read_text("utf-8"));provided=manifest.pop("manifest_sha256",None)
    if manifest.get("format")!=FORMAT or manifest.get("source_sha256")!=source_sha256() or provided!=hashlib.sha256(canonical(manifest)).hexdigest():raise ValueError("trajectory manifest provenance drifted")
    artifact=root/manifest["artifact"]["path"]
    if artifact.stat().st_size!=manifest["artifact"]["bytes"] or hashlib.sha256(artifact.read_bytes()).hexdigest()!=manifest["artifact"]["sha256"]:raise ValueError("trajectory artifact drifted")
    with np.load(artifact,allow_pickle=False) as archive:arrays={name:archive[name] for name in archive.files}
    if set(arrays)!=set(manifest["shapes"]) or any(list(array.shape)!=manifest["shapes"][name] or str(array.dtype)!=manifest["dtypes"][name] for name,array in arrays.items()):raise ValueError("trajectory archive contract drifted")
    if _array_digest(arrays)!=manifest["arrays_sha256"] or len(arrays["tick"])!=manifest["frames"]:raise ValueError("trajectory semantic replay drifted")
    manifest["manifest_sha256"]=provided;return manifest
