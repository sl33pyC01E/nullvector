from __future__ import annotations

import hashlib, json, os, shutil, uuid
from pathlib import Path
import numpy as np

from .contract import ACTIONS, ACTOR_FEATURES, ACTOR_FIELD_SHAPE, ARRAY_NAMES, COUNTERFACTUAL_SHAPE, FORMAT, FRAME_SIZE, ORGANISM_SHAPE, SPATIAL_NAMES, SPATIAL_SHAPE, STATE_FEATURES, canonical, source_sha256

DISK_FLOOR = 100 * 1024**3

def _digest(arrays):
    digest=hashlib.sha256(b"nullvector-whole-viewport-arrays-v5\0")
    for name,array in sorted(arrays.items()):
        contiguous=np.ascontiguousarray(array);digest.update(name.encode()+b"\0"+str(contiguous.dtype).encode()+b"\0"+str(contiguous.shape).encode()+b"\0");digest.update(memoryview(contiguous))
    return digest.hexdigest()

class WholeViewportRecorder:
    def __init__(self, root: Path, *, max_frames=2400):
        if not 64 <= max_frames <= 10000: raise ValueError("whole-viewport frame bound drifted")
        self.root=Path(root);self.max_frames=int(max_frames);self.active=False;self.session_id="";self.world_seed=0;self.start_tick=0;self._rows={name:[] for name in ARRAY_NAMES}
    @property
    def frame_count(self): return len(self._rows["tick"])
    def start(self, session_id, *, world_seed, tick):
        if self.active or not session_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in session_id): raise ValueError("whole-viewport identity drifted")
        if self.root.exists() and shutil.disk_usage(self.root).free < DISK_FLOOR: raise RuntimeError("whole-viewport recorder reached disk floor")
        for value in self._rows.values(): value.clear()
        self.active=True;self.session_id=session_id;self.world_seed=int(world_seed);self.start_tick=int(tick)
    def append(self, **row):
        if not self.active or self.frame_count >= self.max_frames: return False
        if set(row) != set(ARRAY_NAMES): raise ValueError("whole-viewport row members drifted")
        action=row["action"]
        if action not in ACTIONS or int(row["episode_step"]) != self.frame_count: raise ValueError("whole-viewport action/step drifted")
        expected={"frame":((FRAME_SIZE[1],FRAME_SIZE[0],3),np.uint8),"spatial":(SPATIAL_SHAPE,np.float16),"organisms":(ORGANISM_SHAPE,np.float16),"organism_mask":((ORGANISM_SHAPE[0],),np.bool_),"state":((STATE_FEATURES,),np.float32),"actor_state":((ACTOR_FEATURES,),np.float32),"actor_field":(ACTOR_FIELD_SHAPE,np.float16),"visibility":((1,32,32),np.float16),"memory":((1,32,32),np.float16),"control":((4,),np.float32),"timeline":((3,),np.float32),"counterfactual":(COUNTERFACTUAL_SHAPE,np.float32)}
        values={}
        for name,(shape,dtype) in expected.items():
            value=np.asarray(row[name],dtype=dtype)
            if value.shape != shape or (name!="frame" and not np.isfinite(value).all()): raise ValueError(f"whole-viewport {name} drifted")
            values[name]=value.copy()
        if self._rows["tick"] and int(row["tick"]) <= self._rows["tick"][-1]: raise ValueError("whole-viewport episode ticks are not contiguous")
        values.update({"action":ACTIONS.index(action),"selected":int(row["selected"]),"timeline_event":int(row["timeline_event"]),"tick":int(row["tick"]),"episode_step":int(row["episode_step"])})
        for name in ARRAY_NAMES:self._rows[name].append(values[name])
        return True
    def finish(self):
        if not self.active or self.frame_count < 64: raise ValueError("whole-viewport trajectory incomplete")
        self.root.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(self.root).free < DISK_FLOOR: raise RuntimeError("whole-viewport recorder reached disk floor")
        destination=self.root/self.session_id
        if destination.exists(): raise FileExistsError(destination)
        staging=self.root/f".{self.session_id}.tmp-{uuid.uuid4().hex}";staging.mkdir()
        arrays={name:np.stack(self._rows[name]) for name in ("frame","spatial","organisms","organism_mask","state","actor_state","actor_field","visibility","memory","control","timeline","counterfactual")}
        arrays.update({"action":np.asarray(self._rows["action"],np.uint8),"selected":np.asarray(self._rows["selected"],np.int64),"timeline_event":np.asarray(self._rows["timeline_event"],np.uint8),"tick":np.asarray(self._rows["tick"],np.int64),"episode_step":np.asarray(self._rows["episode_step"],np.int32)});arrays={name:arrays[name] for name in ARRAY_NAMES}
        archive=staging/"trajectory.npz";np.savez_compressed(archive,**arrays)
        manifest={"format":FORMAT,"source_sha256":source_sha256(),"session_id":self.session_id,"world_seed":self.world_seed,"start_tick":self.start_tick,"end_tick":int(arrays["tick"][-1]),"frames":self.frame_count,"frame_size":list(FRAME_SIZE),"spatial_channels":list(SPATIAL_NAMES),"actions":list(ACTIONS),"runtime_contract":{"hud_free_view_is_one_vae_decode":True,"no_sprite_tile_or_cell_draw_calls_in_deployment":True,"menus_hud_and_debug_may_use_native_ui":True,"teacher_scaffold_is_not_deployment_renderer":True,"all_conditioning_is_numeric":True},"arrays_sha256":_digest(arrays),"artifact":{"path":archive.name,"bytes":archive.stat().st_size,"sha256":hashlib.sha256(archive.read_bytes()).hexdigest()},"shapes":{name:list(value.shape) for name,value in arrays.items()},"dtypes":{name:str(value.dtype) for name,value in arrays.items()}}
        manifest["manifest_sha256"]=hashlib.sha256(canonical(manifest)).hexdigest();(staging/"manifest.json").write_bytes(canonical(manifest));os.replace(staging,destination);self.active=False;return destination

def validate_trajectory(path: Path):
    root=Path(path);raw=(root/"manifest.json").read_bytes();manifest=json.loads(raw);provided=manifest.pop("manifest_sha256",None)
    if raw != canonical({**manifest,"manifest_sha256":provided}) or manifest.get("format")!=FORMAT or manifest.get("source_sha256")!=source_sha256() or provided!=hashlib.sha256(canonical(manifest)).hexdigest(): raise ValueError("whole-viewport manifest provenance drifted")
    artifact=root/manifest["artifact"]["path"]
    if artifact.stat().st_size!=manifest["artifact"]["bytes"] or hashlib.sha256(artifact.read_bytes()).hexdigest()!=manifest["artifact"]["sha256"]: raise ValueError("whole-viewport artifact drifted")
    with np.load(artifact,allow_pickle=False) as archive: arrays={name:archive[name] for name in archive.files}
    if tuple(arrays)!=ARRAY_NAMES or _digest(arrays)!=manifest["arrays_sha256"]: raise ValueError("whole-viewport semantic replay drifted")
    if any(list(array.shape)!=manifest["shapes"][name] or str(array.dtype)!=manifest["dtypes"][name] for name,array in arrays.items()): raise ValueError("whole-viewport array contract drifted")
    if not np.array_equal(arrays["episode_step"],np.arange(manifest["frames"],dtype=np.int32)) or not np.all(np.diff(arrays["tick"])>0): raise ValueError("whole-viewport sequence drifted")
    manifest["manifest_sha256"]=provided;return manifest
