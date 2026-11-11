"""Threshold arithmetic for a guardrail that hands conversations to a specialist team.

A conversation is handed off when `score >= threshold`. Candidate thresholds are the distinct scores, so every
operating point is one a real threshold can produce: tied scores are handed off together or not at all.

The two questions have opposite tie rules:
  - capacity -> catch rate: the LOWEST threshold whose handoffs stay within capacity (use as much as you have);
  - target catch rate -> capacity: the HIGHEST threshold that still reaches the target (fewest handoffs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


class NoPositivesError(ValueError):
    """Catch rate is undefined when nothing in the data warrants a specialist."""


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float | None  # None means "above every score": no handoffs at all
    handoffs: int
    true_handoffs: int  # handoffs that warrant a specialist
    positives: int
    n: int

    @property
    def catch_rate(self) -> float:
        return self.true_handoffs / self.positives

    @property
    def precision(self) -> float | None:
        return self.true_handoffs / self.handoffs if self.handoffs else None

    @property
    def handoffs_per_1000(self) -> float:
        return 1000.0 * self.handoffs / self.n


def _arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=int)
    if s.shape != y.shape or s.ndim != 1 or len(s) == 0:
        raise ValueError("scores and labels must be non-empty 1-D arrays of the same length")
    if y.sum() == 0:
        raise NoPositivesError("no conversations are labelled as warranting a specialist, so catch rate is undefined")
    return s, y


def _steps(s: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each distinct score, highest first: (threshold, handoffs, true handoffs) if threshold = that score."""
    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    last_of_group = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    idx = np.flatnonzero(last_of_group)
    return s_sorted[idx], idx + 1, np.cumsum(y_sorted)[idx]


def metrics_at_threshold(scores, labels, threshold: float) -> OperatingPoint:
    s, y = _arrays(scores, labels)
    handed = s >= threshold
    return OperatingPoint(float(threshold), int(handed.sum()), int(y[handed].sum()), int(y.sum()), len(s))


def operating_point_for_capacity(scores, labels, capacity: int) -> OperatingPoint:
    """The lowest threshold whose handoffs are <= capacity. A tied group that would overshoot stays out."""
    if capacity < 0:
        raise ValueError("capacity cannot be negative")
    s, y = _arrays(scores, labels)
    thresholds, handoffs, tp = _steps(s, y)
    within = np.flatnonzero(handoffs <= capacity)
    if len(within) == 0:
        return OperatingPoint(None, 0, 0, int(y.sum()), len(s))
    k = within[-1]  # handoffs increase as the threshold falls, so the last one within capacity is the lowest
    return OperatingPoint(float(thresholds[k]), int(handoffs[k]), int(tp[k]), int(y.sum()), len(s))


def positives_needed_for_target(positives: int, target: float) -> int:
    """How many positives must be caught to reach `target`: ceil(target x positives), in integer maths.

    Targets are whole percents (the app's slider). Float maths can add one: 0.56 * 100 is 56.00000000000001,
    whose ceiling is 57. So the target becomes an integer percent first and the ceiling is integer division.
    This counts catches, not handoffs; the two are equal only for a perfect guardrail.
    """
    if not 0.0 < target <= 1.0:
        raise ValueError("target catch rate must be above 0 and at most 1")
    pct = round(target * 100)
    if abs(target * 100 - pct) > 1e-9:
        raise ValueError(f"target catch rate must be a whole percent, got {target!r}")
    return (pct * int(positives) + 99) // 100


def capacity_for_target_catch_rate(scores, labels, target: float) -> OperatingPoint:
    """The highest threshold whose catch rate is >= target, i.e. the fewest handoffs that reach it."""
    positives_needed_for_target(1, target)  # validates the target before touching the data
    s, y = _arrays(scores, labels)
    thresholds, handoffs, tp = _steps(s, y)
    needed = positives_needed_for_target(int(y.sum()), target)
    k = int(np.flatnonzero(tp >= needed)[0])
    return OperatingPoint(float(thresholds[k]), int(handoffs[k]), int(tp[k]), int(y.sum()), len(s))


def perfect_guardrail_need(labels, target: float) -> int:
    """The fewest handoffs that could reach `target` if every handoff were a true one: the hard floor on capacity."""
    y = np.asarray(labels, dtype=int)
    if y.sum() == 0:
        raise NoPositivesError("no conversations are labelled as warranting a specialist, so catch rate is undefined")
    return positives_needed_for_target(int(y.sum()), target)


def max_catch_rate_at_capacity(labels, capacity: int) -> float:
    """The best catch rate any guardrail could reach with `capacity` handoffs: min(capacity, positives) / positives."""
    if capacity < 0:
        raise ValueError("capacity cannot be negative")
    y = np.asarray(labels, dtype=int)
    positives = int(y.sum())
    if positives == 0:
        raise NoPositivesError("no conversations are labelled as warranting a specialist, so catch rate is undefined")
    return min(capacity, positives) / positives


def per_1000_to_count(per_1000: float, n: int) -> int:
    """Convert a capacity per 1,000 conversations to a handoff count, rounding down to stay within it."""
    return int(math.floor(per_1000 * n / 1000.0 + 1e-9))


def shortfall_per_1000(required_per_1000: float, staffed_per_1000: float) -> float:
    """How far the need exceeds staffed capacity; 0 when it doesn't. The flag is `shortfall > 0`."""
    return max(0.0, required_per_1000 - staffed_per_1000)


def misses_by_characteristic(df: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    """Positives the threshold does not hand off, by characteristic."""
    pos = df[df["warrants_specialist"] == 1]
    if threshold is None:
        missed = pd.Series(True, index=pos.index)
    else:
        missed = pos["score"] < threshold
    out = (
        pd.DataFrame({"characteristic": pos["characteristic"], "missed": missed})
        .groupby("characteristic", sort=True)
        .agg(positives=("missed", "size"), missed=("missed", "sum"))
        .reset_index()
    )
    out["missed"] = out["missed"].astype(int)
    out["missed_share"] = out["missed"] / out["positives"]
    return out


def curve(scores, labels) -> pd.DataFrame:
    """Catch rate and precision at every distinct threshold, for the chart."""
    s, y = _arrays(scores, labels)
    thresholds, handoffs, tp = _steps(s, y)
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "handoffs_per_1000": 1000.0 * handoffs / len(s),
            "catch_rate": tp / y.sum(),
            "precision": tp / handoffs,
        }
    )
