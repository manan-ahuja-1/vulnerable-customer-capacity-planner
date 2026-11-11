"""Validate an uploaded CSV of labelled guardrail scores.

Schema (one row per conversation):
  conversation_id      unique text
  score                a number from 0 to 1
  warrants_specialist  0 or 1: does this conversation warrant a specialist route?
  characteristic       health | life_events | resilience | capability | unknown when warrants_specialist is 1;
                       blank when it is 0

Rows are numbered from 1, counting data rows only (the header is not row 1).
"""

from __future__ import annotations

import pandas as pd

from .synth import CHARACTERISTICS, COLUMNS

ALLOWED_CHARACTERISTICS = (*CHARACTERISTICS, "unknown")
MAX_LISTED_ERRORS = 20


def validate_upload(df: pd.DataFrame) -> tuple[pd.DataFrame | None, list[str]]:
    """Return (clean frame, []) or (None, errors). Read the CSV with dtype=str and keep_default_na=False."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return None, [f"missing column(s): {', '.join(missing)}. Expected: {', '.join(COLUMNS)}"]
    if df.empty:
        return None, ["the file has a header but no rows"]

    df = df[COLUMNS].astype(str).apply(lambda col: col.str.strip())
    errors: list[str] = []

    blank_id = df.index[df["conversation_id"] == ""]
    errors += [f"row {i + 1}: conversation_id is blank" for i in blank_id]
    dupes = df["conversation_id"][(df["conversation_id"] != "") & df["conversation_id"].duplicated(keep=False)]
    for cid in dict.fromkeys(dupes):
        rows = ", ".join(str(i + 1) for i in df.index[df["conversation_id"] == cid])
        errors.append(f"conversation_id {cid!r} appears more than once (rows {rows})")

    score = pd.to_numeric(df["score"], errors="coerce")
    for i in df.index[score.isna()]:
        errors.append(f"row {i + 1}: score {df.at[i, 'score']!r} is not a number")
    for i in df.index[score.notna() & ((score < 0) | (score > 1))]:
        errors.append(f"row {i + 1}: score {score[i]} is outside 0 to 1")

    label = df["warrants_specialist"].replace({"0.0": "0", "1.0": "1"})
    for i in df.index[~label.isin(["0", "1"])]:
        errors.append(f"row {i + 1}: warrants_specialist {df.at[i, 'warrants_specialist']!r} must be 0 or 1")

    char = df["characteristic"].str.lower()
    pos, neg = label == "1", label == "0"
    for i in df.index[pos & ~char.isin(ALLOWED_CHARACTERISTICS)]:
        errors.append(
            f"row {i + 1}: characteristic {df.at[i, 'characteristic']!r} must be one of "
            f"{', '.join(ALLOWED_CHARACTERISTICS)} when warrants_specialist is 1"
        )
    for i in df.index[neg & (char != "")]:
        errors.append(f"row {i + 1}: characteristic must be blank when warrants_specialist is 0")

    if not errors and not pos.any():
        errors.append("no rows have warrants_specialist = 1, so catch rate cannot be computed")

    if errors:
        extra = len(errors) - MAX_LISTED_ERRORS
        return None, errors[:MAX_LISTED_ERRORS] + ([f"... and {extra} more"] if extra > 0 else [])

    clean = pd.DataFrame(
        {
            "conversation_id": df["conversation_id"],
            "score": score.astype(float),
            "warrants_specialist": label.astype(int),
            "characteristic": char,
        }
    )
    return clean, []
