"""Test 5: upload validation gives a specific, row-level message for each kind of fault."""

from pathlib import Path

import pandas as pd
import pytest

from src.schema import MAX_LISTED_ERRORS, validate_upload

ROOT = Path(__file__).resolve().parents[1]


def frame(rows):
    return pd.DataFrame(rows, columns=["conversation_id", "score", "warrants_specialist", "characteristic"]).astype(str)


GOOD = [["a", "0.9", "1", "health"], ["b", "0.2", "0", ""], ["c", "0.5", "1", "unknown"]]


def test_valid_frame_passes_and_is_typed():
    clean, errors = validate_upload(frame(GOOD))
    assert errors == []
    assert clean["score"].dtype == float and clean["warrants_specialist"].tolist() == [1, 0, 1]


def test_committed_sample_passes():
    raw = pd.read_csv(ROOT / "data" / "sample_synthetic.csv", dtype=str, keep_default_na=False)
    clean, errors = validate_upload(raw)
    assert errors == [] and len(clean) == 2_000


def test_missing_column():
    _, errors = validate_upload(frame(GOOD).drop(columns="score"))
    assert errors and "missing column(s): score" in errors[0]


@pytest.mark.parametrize(
    "row, expected",
    [
        (["d", "1.3", "0", ""], "row 4: score 1.3 is outside 0 to 1"),
        (["d", "high", "0", ""], "row 4: score 'high' is not a number"),
        (["d", "0.4", "yes", ""], "row 4: warrants_specialist 'yes' must be 0 or 1"),
        (["d", "0.4", "1", "grief"], "row 4: characteristic 'grief' must be one of"),
        (["d", "0.4", "0", "health"], "row 4: characteristic must be blank when warrants_specialist is 0"),
        (["", "0.4", "0", ""], "row 4: conversation_id is blank"),
        (["a", "0.4", "0", ""], "conversation_id 'a' appears more than once (rows 1, 4)"),
    ],
)
def test_each_fault_has_a_specific_message(row, expected):
    clean, errors = validate_upload(frame(GOOD + [row]))
    assert clean is None
    assert any(e.startswith(expected) for e in errors), errors


def test_no_positives():
    _, errors = validate_upload(frame([["a", "0.9", "0", ""], ["b", "0.2", "0", ""]]))
    assert errors == ["no rows have warrants_specialist = 1, so catch rate cannot be computed"]


def test_error_list_is_capped():
    rows = GOOD + [[f"x{i}", "2", "0", ""] for i in range(30)]
    _, errors = validate_upload(frame(rows))
    assert len(errors) == MAX_LISTED_ERRORS + 1 and errors[-1] == "... and 10 more"
