"""The MI export: one markdown table, Measure | Value | Basis, plus one footnote line."""

from __future__ import annotations

import pandas as pd

from .basis import ASSUMED, result_basis
from .calibrate import OperatingPoint
from .synth import CHARACTERISTIC_LABELS


SAMPLING_WARNING = (
    "Catch rate is only valid if your sample includes conversations the guardrail didn't hand off. "
    "A file of handoffs alone will overstate it."
)


def footnote_for(source: str, n: int, seed: int) -> str:
    """The line under the MI table. Uploads carry the sampling warning, because every result rests on the labels."""
    if source == "demo":
        return (
            f"Data: {n:,} synthetic conversations generated from stated assumptions (seed {seed}). "
            "Not a finding about real traffic."
        )
    if source == "upload":
        return f"Data: uploaded file, {n:,} conversations. Results computed from your data. {SAMPLING_WARNING}"
    raise ValueError(f"source must be 'demo' or 'upload', got {source!r}")


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def mi_rows(
    target: float,
    point: OperatingPoint,
    staffed_per_1000: float,
    shortfall: float,
    misses: pd.DataFrame,
    source: str,
) -> list[tuple[str, str, str]]:
    r = result_basis(source)
    rows = [
        ("Target catch rate", _pct(target), ASSUMED),
        ("Required specialist capacity (handoffs per 1,000)", f"{point.handoffs_per_1000:.1f}", r),
        ("Staffed capacity (handoffs per 1,000)", f"{staffed_per_1000:.1f}", ASSUMED),
        ("Shortfall (handoffs per 1,000)", f"{shortfall:.1f}", r),
        ("Threshold (hand off when score ≥)", "above all scores" if point.threshold is None else f"{point.threshold:.4f}", r),
        ("Precision at this threshold", _pct(point.precision), r),
        ("Catch rate at this threshold", _pct(point.catch_rate), r),
    ]
    for row in misses.itertuples():
        name = CHARACTERISTIC_LABELS.get(row.characteristic, row.characteristic)
        rows.append((f"Missed: {name}", f"{row.missed} of {row.positives}", r))
    return rows


def mi_table_markdown(rows: list[tuple[str, str, str]], footnote: str) -> str:
    lines = ["| Measure | Value | Basis |", "|---|---|---|"]
    lines += [f"| {m} | {v} | {b} |" for m, v, b in rows]
    return "\n".join(lines) + f"\n\n{footnote}\n"
