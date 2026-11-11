"""Test 7: the MI export is exactly one markdown table, with every row tagged by its basis."""

import re

import pandas as pd

from src import calibrate as cal
from src.basis import ASSUMED, FROM_YOUR_DATA, SYNTHETIC
from src.report import SAMPLING_WARNING, footnote_for, mi_rows, mi_table_markdown

SCORES = [0.95, 0.90, 0.80, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.10]
LABELS = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
CHARS = ["health", "", "life_events", "capability", "", "capability", "", "", "resilience", ""]
DF = pd.DataFrame({"score": SCORES, "warrants_specialist": LABELS, "characteristic": CHARS})


def build(source):
    need = cal.capacity_for_target_catch_rate(SCORES, LABELS, 0.8)
    short = cal.shortfall_per_1000(need.handoffs_per_1000, 500.0)
    rows = mi_rows(0.8, need, 500.0, short, cal.misses_by_characteristic(DF, need.threshold), source)
    return rows, mi_table_markdown(rows, "Data: footnote line.")


def test_exactly_one_table_with_footnote():
    _, md = build("demo")
    assert len([ln for ln in md.splitlines() if re.fullmatch(r"\|(-{3}\|)+", ln)]) == 1
    assert md.splitlines()[0] == "| Measure | Value | Basis |"
    assert md.rstrip().endswith("Data: footnote line.")


def test_demo_tags_inputs_assumed_and_results_synthetic():
    rows, _ = build("demo")
    basis = {m: b for m, _, b in rows}
    assert basis["Target catch rate"] == ASSUMED
    assert basis["Staffed capacity (handoffs per 1,000)"] == ASSUMED
    assert {b for m, _, b in rows if m not in ("Target catch rate", "Staffed capacity (handoffs per 1,000)")} == {SYNTHETIC}


def test_upload_tags_results_from_your_data():
    rows, _ = build("upload")
    assert SYNTHETIC not in {b for _, _, b in rows}
    assert FROM_YOUR_DATA in {b for _, _, b in rows}


def test_values_match_hand_computation():
    rows, _ = build("demo")
    v = {m: val for m, val, _ in rows}
    assert v["Required specialist capacity (handoffs per 1,000)"] == "600.0"  # 6 of 10 handoffs
    assert v["Precision at this threshold"] == "66.7%"
    assert v["Missed: Resilience"] == "1 of 1"


# The sampling warning on uploads (added 11 Nov 2026) -------------------------------------------------------------
def test_upload_footnote_carries_the_sampling_warning():
    note = footnote_for("upload", 2000, 7)
    assert SAMPLING_WARNING in note
    md = mi_table_markdown([("Target catch rate", "90.0%", "assumed")], note)
    assert md.count("| Measure | Value | Basis |") == 1
    assert md.rstrip().endswith(SAMPLING_WARNING)


def test_demo_footnote_has_no_sampling_warning():
    note = footnote_for("demo", 20000, 7)
    assert SAMPLING_WARNING not in note
    assert "synthetic" in note
