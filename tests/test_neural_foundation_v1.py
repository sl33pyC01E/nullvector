from __future__ import annotations
from forge.neural_foundation_v1.contract import COMPONENTS,REQUIRED_DOMAINS
from forge.neural_foundation_v1.registry import _report_ready
def test_foundation_specs_cover_every_scale()->None:
    assert set(REQUIRED_DOMAINS)<={row[1] for row in COMPONENTS};assert len({row[0] for row in COMPONENTS})==len(COMPONENTS)
def test_ready_evidence_is_fail_closed()->None:
    assert _report_ready({"gates":{"a":True,"b":True}});assert not _report_ready({"gates":{"a":True,"b":False}});assert not _report_ready({})
