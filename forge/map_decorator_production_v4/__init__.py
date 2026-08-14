"""Public-entropy proposal substrate for hybrid neural map decoration v4."""

from .contract import V4_CONTRACT_SHA256, ProposalLocatorConfig
from .proposal import ProposalAuthority, ProposalFields, audit_proposal_targets, build_proposal_fields
from .model import ProposalConditionedDecoratorV4, ProposalLocatorOutputV4
from .decoding import select_proposal_conditioned_argmax
from .smoke import build_smoke, validate_smoke
from .audit import build_full_audit, validate_full_audit

__all__ = [
    "ProposalAuthority",
    "ProposalFields",
    "ProposalLocatorConfig",
    "ProposalConditionedDecoratorV4",
    "ProposalLocatorOutputV4",
    "V4_CONTRACT_SHA256",
    "audit_proposal_targets",
    "build_proposal_fields",
    "select_proposal_conditioned_argmax",
    "build_smoke",
    "validate_smoke",
    "build_full_audit",
    "validate_full_audit",
]
