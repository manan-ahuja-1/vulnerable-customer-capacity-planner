"""Test 6: the generator against its own configuration. This checks the generator, not any evaluation."""

import pandas as pd
import pytest

from src.synth import CHARACTERISTICS, COLUMNS, Assumptions, ScoreDist, allocate, generate


def test_same_seed_same_frame():
    pd.testing.assert_frame_equal(generate(Assumptions(), seed=7), generate(Assumptions(), seed=7))


def test_different_seed_different_scores():
    assert not generate(Assumptions(), seed=7)["score"].equals(generate(Assumptions(), seed=8)["score"])


def test_counts_match_the_assumptions_exactly():
    df = generate(Assumptions(), seed=7)  # 20,000 rows, 6% prevalence, equal mix
    assert list(df.columns) == COLUMNS and len(df) == 20_000
    assert df["warrants_specialist"].sum() == 1_200
    pos = df[df["warrants_specialist"] == 1]
    assert pos["characteristic"].value_counts().to_dict() == {c: 300 for c in CHARACTERISTICS}
    assert (df.loc[df["warrants_specialist"] == 0, "characteristic"] == "").all()
    assert df["conversation_id"].is_unique
    assert df["score"].between(0, 1).all()


def test_positive_score_means_follow_their_assumptions():
    a = Assumptions(positive_scores={c: ScoreDist(m, "medium") for c, m in zip(CHARACTERISTICS, (0.8, 0.7, 0.6, 0.5))})
    means = generate(a, seed=7).query("warrants_specialist == 1").groupby("characteristic")["score"].mean()
    for c, m in zip(CHARACTERISTICS, (0.8, 0.7, 0.6, 0.5)):
        assert means[c] == pytest.approx(m, abs=0.03)


def test_allocate_is_exact():
    assert allocate(10, {"a": 1, "b": 1, "c": 1}) == {"a": 4, "b": 3, "c": 3}  # equal remainders: alphabetical
    assert sum(allocate(1_201, {"a": 3, "b": 1, "c": 1, "d": 2}).values()) == 1_201
    with pytest.raises(ValueError):
        allocate(10, {"a": 0, "b": 0})


@pytest.mark.parametrize(
    "bad",
    [
        dict(prevalence=0.0),
        dict(prevalence=1.0),
        dict(hard_negative_share=1.5),
        dict(routine_scores=ScoreDist(1.0)),
        dict(routine_scores=ScoreDist(0.5, "huge")),
    ],
)
def test_invalid_assumptions_raise(bad):
    with pytest.raises(ValueError):
        generate(Assumptions(**bad), seed=7)


def test_json_round_trip():
    a = Assumptions(prevalence=0.1, positive_scores={c: ScoreDist(0.6, "low") for c in CHARACTERISTICS})
    assert Assumptions.from_json(a.to_json()) == a
