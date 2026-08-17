from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from .contract import ACTIONS, ACTOR_FEATURES, ACTOR_FIELD_SHAPE, ARRAY_NAMES, COUNTERFACTUAL_SHAPE, FORMAT, FRAME_SIZE, STATE_FEATURES, canonical, source_sha256


DISK_FLOOR = 100 * 1024**3


def _array_digest(arrays):
    digest = hashlib.sha256(b"nullvector-natural-play-cellular-arrays-v4\0")
    for name, array in sorted(arrays.items()):
        digest.update(name.encode() + b"\0" + str(array.dtype).encode() + b"\0" + str(array.shape).encode() + b"\0" + array.tobytes())
    return digest.hexdigest()


class NaturalPlayRecorder:
    def __init__(self, root: Path, *, max_frames=2400):
        if not 128 <= max_frames <= 10000:raise ValueError("natural teacher frame bound drifted")
        self.root=Path(root);self.max_frames=int(max_frames);self.active=False;self.session_id="";self.world_seed=0;self.start_tick=0;self._rows={name:[] for name in ARRAY_NAMES}

    @property
    def frame_count(self):return len(self._rows["tick"])

    def start(self, session_id, *, world_seed, tick):
        if self.active or not session_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in session_id):raise ValueError("natural teacher identity drifted")
        if self.root.exists() and shutil.disk_usage(self.root).free < DISK_FLOOR:raise RuntimeError("natural teacher reached disk floor")
        for value in self._rows.values():value.clear()
        self.active=True;self.session_id=session_id;self.world_seed=int(world_seed);self.start_tick=int(tick)

    def append(self, *, frame, state, actor_state, actor_field, control, action, selected, timeline_event, timeline, counterfactual, tick, episode_step):
        if not self.active or self.frame_count >= self.max_frames:return False
        if action not in ACTIONS or int(episode_step) != self.frame_count:raise ValueError("natural teacher action/step drifted")
        values=(np.asarray(frame),np.asarray(state,np.float32),np.asarray(actor_state,np.float32),np.asarray(actor_field,np.float16),np.asarray(control,np.float32),np.asarray(timeline,np.float32),np.asarray(counterfactual,np.float32));frame,state,actor_state,actor_field,control,timeline,counterfactual=values
        if frame.shape != (FRAME_SIZE[1],FRAME_SIZE[0],3) or frame.dtype != np.uint8:raise ValueError("natural teacher frame drifted")
        if state.shape != (STATE_FEATURES,) or actor_state.shape != (ACTOR_FEATURES,) or actor_field.shape != ACTOR_FIELD_SHAPE or control.shape != (4,) or timeline.shape != (3,) or counterfactual.shape != COUNTERFACTUAL_SHAPE:raise ValueError("natural teacher tensor shape drifted")
        if not all(np.isfinite(value).all() for value in values[1:]):raise ValueError("natural teacher nonfinite tensor")
        if self._rows["tick"] and int(tick) <= self._rows["tick"][-1]:raise ValueError("natural teacher ticks are not strictly contiguous")
        row=(frame.copy(),state.copy(),actor_state.copy(),actor_field.copy(),control.copy(),ACTIONS.index(action),int(selected),int(timeline_event),timeline.copy(),counterfactual.copy(),int(tick),int(episode_step))
        for name,value in zip(ARRAY_NAMES,row):self._rows[name].append(value)
        return True

    def finish(self):
        if not self.active or self.frame_count < 128:raise ValueError("natural teacher trajectory is incomplete")
        self.root.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(self.root).free < DISK_FLOOR:raise RuntimeError("natural teacher reached disk floor")
        destination=self.root/self.session_id
        if destination.exists():raise FileExistsError(destination)
        staging=self.root/f".{self.session_id}.tmp-{uuid.uuid4().hex}";staging.mkdir()
        arrays={name:np.stack(self._rows[name]) for name in ("frame","state","actor_state","actor_field","control","timeline","counterfactual")};arrays.update({"action":np.asarray(self._rows["action"],np.uint8),"selected":np.asarray(self._rows["selected"],np.int64),"timeline_event":np.asarray(self._rows["timeline_event"],np.uint8),"tick":np.asarray(self._rows["tick"],np.int64),"episode_step":np.asarray(self._rows["episode_step"],np.int32)});arrays={name:arrays[name] for name in ARRAY_NAMES};archive=staging/"trajectory.npz";np.savez_compressed(archive,**arrays)
        manifest={"format":FORMAT,"source_sha256":source_sha256(),"session_id":self.session_id,"world_seed":self.world_seed,"start_tick":self.start_tick,"end_tick":int(arrays["tick"][-1]),"frames":self.frame_count,"frame_size":list(FRAME_SIZE),"actions":list(ACTIONS),"view_contract":{"clean_student_view":True,"diagnostic_overlays":False,"fixed_actor_camera":True,"no_staged_camera_cuts":True,"teacher_capture_precedes_overlay_composite":True},"arrays_sha256":_array_digest(arrays),"artifact":{"path":archive.name,"bytes":archive.stat().st_size,"sha256":hashlib.sha256(archive.read_bytes()).hexdigest()},"shapes":{name:list(array.shape) for name,array in arrays.items()},"dtypes":{name:str(array.dtype) for name,array in arrays.items()}};manifest["manifest_sha256"]=hashlib.sha256(canonical(manifest)).hexdigest();(staging/"manifest.json").write_bytes(canonical(manifest));os.replace(staging,destination);self.active=False;return destination


def validate_trajectory(path: Path):
    root=Path(path);raw=(root/"manifest.json").read_bytes();manifest=json.loads(raw);provided=manifest.pop("manifest_sha256",None)
    if raw != canonical({**manifest,"manifest_sha256":provided}) or manifest.get("format") != FORMAT or manifest.get("source_sha256") != source_sha256() or provided != hashlib.sha256(canonical(manifest)).hexdigest():raise ValueError("natural teacher manifest provenance drifted")
    expected={"clean_student_view":True,"diagnostic_overlays":False,"fixed_actor_camera":True,"no_staged_camera_cuts":True,"teacher_capture_precedes_overlay_composite":True}
    if manifest.get("view_contract") != expected:raise ValueError("natural teacher view contract drifted")
    artifact=root/manifest["artifact"]["path"]
    if artifact.stat().st_size != manifest["artifact"]["bytes"] or hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest["artifact"]["sha256"]:raise ValueError("natural teacher artifact drifted")
    with np.load(artifact,allow_pickle=False) as archive:arrays={name:archive[name] for name in archive.files}
    if tuple(arrays) != ARRAY_NAMES or set(arrays) != set(manifest["shapes"]):raise ValueError("natural teacher members drifted")
    if any(list(array.shape) != manifest["shapes"][name] or str(array.dtype) != manifest["dtypes"][name] for name,array in arrays.items()):raise ValueError("natural teacher array contract drifted")
    if _array_digest(arrays) != manifest["arrays_sha256"] or len(arrays["tick"]) != manifest["frames"] or len(arrays["tick"]) < 128 or not np.all(np.diff(arrays["tick"]) > 0) or not np.array_equal(arrays["episode_step"],np.arange(len(arrays["tick"]),dtype=np.int32)):raise ValueError("natural teacher semantic replay drifted")
    if np.unique(arrays["selected"]).size != 1:raise ValueError("natural teacher camera actor drifted")
    manifest["manifest_sha256"]=provided;return manifest
