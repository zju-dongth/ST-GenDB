"""
ST-GenDB: Admission-Gated Plan Synthesis and Progressive Token Buffering for Multi-Camera Video.
"""

from .schema import (
    CameraTopology,
    CompetitiveProvenanceView,
    OperatorType,
    PhysicalOperatorNode,
    PhysicalPlanDAG,
    ProvenanceEdge,
    TokenResolutionTier,
    VisualDetectionSlice,
)
from .admission_gate import (
    AdmissionAuditReport,
    ClauseAdmissionGate,
    theoretical_coupled_provenance_recall,
    theoretical_recall_collapse_top1,
)
from .buffer_pool import BufferPoolTelemetry, BufferSlotEntry, ProgressiveVisualBufferPool
from .algebra import SpatioTemporalSemanticAlgebra, estimate_empirical_tier0_hubs
from .calibration import (
    PredicateCalibrator,
    SubspaceProjector,
    fit_multitier_subspace,
    fit_predicate_calibration,
)
from .synthesizer import DeclarativeVideoQuery, TopologyAwarePlanSynthesizer
from .pipeline import QueryExecutionResult, STGenDBEngine, compute_f1
from .vlm_backend import Qwen25VLHardwareBackend, VisualKVSlot, build_tier_image

__all__ = [
    "CameraTopology",
    "CompetitiveProvenanceView",
    "OperatorType",
    "PhysicalOperatorNode",
    "PhysicalPlanDAG",
    "ProvenanceEdge",
    "TokenResolutionTier",
    "VisualDetectionSlice",
    "AdmissionAuditReport",
    "ClauseAdmissionGate",
    "theoretical_coupled_provenance_recall",
    "theoretical_recall_collapse_top1",
    "BufferPoolTelemetry",
    "BufferSlotEntry",
    "ProgressiveVisualBufferPool",
    "SpatioTemporalSemanticAlgebra",
    "estimate_empirical_tier0_hubs",
    "PredicateCalibrator",
    "SubspaceProjector",
    "fit_multitier_subspace",
    "fit_predicate_calibration",
    "DeclarativeVideoQuery",
    "TopologyAwarePlanSynthesizer",
    "QueryExecutionResult",
    "STGenDBEngine",
    "compute_f1",
    "Qwen25VLHardwareBackend",
    "VisualKVSlot",
    "build_tier_image",
]
