"""Nonconformity scores used in the experiments."""

from __future__ import annotations

from socp.conformal.scores.absres import AbsoluteResidualScore
from socp.conformal.scores.chr import CHRScore
from socp.conformal.scores.classification import (
    APSScore,
    LabelRankingScore,
    LACScore,
    RAPSScore,
    SAPSScore,
)
from socp.conformal.scores.cqr import CQRScore
from socp.conformal.scores.flow import NormalizingFlowScore
from socp.conformal.scores.rectified import RectifiedScore

__all__ = (
    "AbsoluteResidualScore",
    "CQRScore",
    "CHRScore",
    "NormalizingFlowScore",
    "RectifiedScore",
    "LACScore",
    "APSScore",
    "RAPSScore",
    "SAPSScore",
    "LabelRankingScore",
)
