"""Metrics and paired statistics used in the experiments."""

from socp.evaluation.conditional import (
    excess_risk_of_target,
    size_stratified_coverage,
    worst_slab_coverage,
)
from socp.evaluation.metrics import (
    CoverageGaps,
    PredictionMetrics,
    ScalarMetric,
    coverage_gaps,
    prediction_metrics,
)
from socp.evaluation.statistics import (
    PairedComparison,
    holm,
    paired_comparison,
)

__all__ = (
    "CoverageGaps",
    "PairedComparison",
    "PredictionMetrics",
    "ScalarMetric",
    "coverage_gaps",
    "excess_risk_of_target",
    "holm",
    "paired_comparison",
    "prediction_metrics",
    "size_stratified_coverage",
    "worst_slab_coverage",
)
