from __future__ import annotations

import numpy as np
import pytest

from forge.recurrent_world_runtime_v1.runtime import RecurrentWorldRuntime


class _FakeStream:
    pass


def test_runtime_input_contracts_fail_closed_without_loading_weights():
    frame=np.zeros((256,256,3),np.uint8)
    assert RecurrentWorldRuntime._frame(frame).flags.c_contiguous
    with pytest.raises(ValueError,match="world frame"):RecurrentWorldRuntime._frame(np.zeros((64,64,3),np.uint8))
    np.testing.assert_array_equal(RecurrentWorldRuntime._vector(np.zeros(4),4,"control"),np.zeros(4,np.float32))
    with pytest.raises(ValueError,match="control"):RecurrentWorldRuntime._vector(np.zeros(5),4,"control")
    broken=np.zeros(4,np.float32);broken[2]=np.nan
    with pytest.raises(ValueError,match="finite"):RecurrentWorldRuntime._vector(broken,4,"control")


def test_forecast_schedule_validation_is_bounded_before_stream_creation():
    runtime=RecurrentWorldRuntime.__new__(RecurrentWorldRuntime);runtime.stream=lambda *_args,**_kwargs:_FakeStream()
    frame=np.zeros((256,256,3),np.uint8);actor=np.zeros(128,np.float32);control=np.zeros(4,np.float32);state=np.zeros(64,np.float32)
    with pytest.raises(ValueError,match="horizon"):runtime.forecast(frame,actor,actions=[0],controls=control,states=state,horizon=0)
    with pytest.raises(ValueError,match="action schedule"):runtime.forecast(frame,actor,actions=[0,1],controls=control,states=state,horizon=3)
    with pytest.raises(ValueError,match="control schedule"):runtime.forecast(frame,actor,actions=[0],controls=np.zeros((2,4),np.float32),states=state,horizon=3)


def test_release_loads_and_continuous_stream_advances_on_cpu():
    runtime=RecurrentWorldRuntime.from_release(device="cpu")
    frame=np.zeros((256,256,3),np.uint8);frame[96:160,104:152]=(30,180,210);actor=np.zeros(128,np.float32);control=np.asarray((.25,0,0,0),np.float32);state=np.zeros(64,np.float32)
    stream=runtime.stream(frame,actor);first=stream.advance(0,control,state);second=stream.advance(0,control,state)
    assert first.index==1 and second.index==2 and first.frame.shape==(256,256,3) and second.actor_state.shape==(128,)
    assert np.isfinite(second.actor_state).all() and runtime.parameter_count==40_087_472
