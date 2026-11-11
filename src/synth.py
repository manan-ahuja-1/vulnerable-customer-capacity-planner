"""Synthetic demo data: one row per conversation, with a guardrail score and a label.

Every number in `Assumptions` is an assumption, not a measurement. The generator draws scores from Beta
distributions, each set by a mean and a spread, so an operator can say "positives score about 0.7" without
thinking in alpha and beta.

Counts are allocated exactly (largest remainder), not sampled, so the realised prevalence and mix always equal
the assumptions. Only the scores are random.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CHARACTERISTICS = ("health", "life_events", "resilience", "capability")
CHARACTERISTIC_LABELS = {
    "health": "Health",
    "life_events": "Life events",
    "resilience": "Resilience",
    "capability": "Capability",
    "unknown": "Unknown",
}

# Beta concentration (alpha + beta) for each spread. Higher concentration means scores cluster tighter.
SPREADS = {"low": 30.0, "medium": 10.0, "high": 4.0}

COLUMNS = ["conversation_id", "score", "warrants_specialist", "characteristic"]


@dataclass
class ScoreDist:
    mean: float
    spread: str = "medium"

    def alpha_beta(self) -> tuple[float, float]:
        if not 0.0 < self.mean < 1.0:
            raise ValueError(f"mean must be between 0 and 1 (exclusive), got {self.mean}")
        if self.spread not in SPREADS:
            raise ValueError(f"spread must be one of {sorted(SPREADS)}, got {self.spread!r}")
        k = SPREADS[self.spread]
        return self.mean * k, (1.0 - self.mean) * k


def _equal_mix() -> dict[str, float]:
    return {c: 0.25 for c in CHARACTERISTICS}


def _equal_positive_scores() -> dict[str, ScoreDist]:
    # Equal across characteristics by default, so no pattern is built in.
    return {c: ScoreDist(0.70, "medium") for c in CHARACTERISTICS}


@dataclass
class Assumptions:
    """Every field is an assumption. The app tags each one 'assumed' on screen."""

    n_conversations: int = 20_000
    prevalence: float = 0.06  # share of conversations that warrant a specialist route
    characteristic_mix: dict[str, float] = field(default_factory=_equal_mix)  # weights, normalised
    positive_scores: dict[str, ScoreDist] = field(default_factory=_equal_positive_scores)
    hard_negative_share: float = 0.25  # of negatives: ordinary hardship and complaints
    hard_negative_scores: ScoreDist = field(default_factory=lambda: ScoreDist(0.45, "medium"))
    routine_scores: ScoreDist = field(default_factory=lambda: ScoreDist(0.10, "medium"))

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Assumptions":
        d = json.loads(s)
        d["positive_scores"] = {k: ScoreDist(**v) for k, v in d["positive_scores"].items()}
        d["hard_negative_scores"] = ScoreDist(**d["hard_negative_scores"])
        d["routine_scores"] = ScoreDist(**d["routine_scores"])
        return cls(**d)


def allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split `total` across keys in proportion to `weights`, exactly (largest-remainder method)."""
    w = {k: float(v) for k, v in weights.items()}
    if any(v < 0 for v in w.values()) or sum(w.values()) <= 0:
        raise ValueError("characteristic mix weights must be non-negative and not all zero")
    s = sum(w.values())
    raw = {k: total * v / s for k, v in w.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    left = total - sum(counts.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - counts[k]), k))[:left]:  # biggest remainder; ties alphabetical
        counts[k] += 1
    return counts


def generate(a: Assumptions, seed: int = 7) -> pd.DataFrame:
    """Return a frame with COLUMNS. `characteristic` is '' for conversations that don't warrant a specialist."""
    if a.n_conversations < 1:
        raise ValueError("n_conversations must be at least 1")
    if not 0.0 < a.prevalence < 1.0:
        raise ValueError(f"prevalence must be between 0 and 1 (exclusive), got {a.prevalence}")
    if not 0.0 <= a.hard_negative_share <= 1.0:
        raise ValueError(f"hard_negative_share must be between 0 and 1, got {a.hard_negative_share}")

    rng = np.random.default_rng(seed)
    n_pos = int(round(a.n_conversations * a.prevalence))
    n_neg = a.n_conversations - n_pos
    n_hard = int(round(n_neg * a.hard_negative_share))

    scores, labels, chars = [], [], []
    for c, k in allocate(n_pos, a.characteristic_mix).items():
        scores.append(rng.beta(*a.positive_scores[c].alpha_beta(), size=k))
        labels.append(np.ones(k, dtype=int))
        chars += [c] * k
    for dist, k in ((a.hard_negative_scores, n_hard), (a.routine_scores, n_neg - n_hard)):
        scores.append(rng.beta(*dist.alpha_beta(), size=k))
        labels.append(np.zeros(k, dtype=int))
        chars += [""] * k

    df = pd.DataFrame(
        {
            "score": np.round(np.concatenate(scores), 4),
            "warrants_specialist": np.concatenate(labels),
            "characteristic": chars,
        }
    )
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df.insert(0, "conversation_id", [f"c{i:05d}" for i in range(1, len(df) + 1)])
    return df[COLUMNS]


def main() -> None:
    p = argparse.ArgumentParser(description="Write the committed sample CSV from the default assumptions.")
    p.add_argument("--write-sample", type=Path, required=True)
    p.add_argument("--rows", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    generate(Assumptions(n_conversations=args.rows), seed=args.seed).to_csv(args.write_sample, index=False)
    print(f"wrote {args.write_sample} ({args.rows} rows, seed {args.seed})")


if __name__ == "__main__":
    main()
