"""Vulnerable customer capacity planner: a thin Streamlit UI over src/. All arithmetic lives in src/calibrate.py."""

from __future__ import annotations

import tomllib
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import calibrate as cal
from src.basis import ASSUMED, label, result_basis
from src.report import SAMPLING_WARNING, footnote_for, mi_rows, mi_table_markdown
from src.schema import ALLOWED_CHARACTERISTICS, validate_upload
from src.synth import CHARACTERISTIC_LABELS, CHARACTERISTICS, SPREADS, Assumptions, ScoreDist, generate

ROOT = Path(__file__).resolve().parent
PITCH = tomllib.loads((ROOT / "pitch.toml").read_text(encoding="utf-8"))
FIRM = PITCH["firm"]["name"]

# Reference palette (dataviz skill): one series, secondary ink for reference lines, recessive grid.
SERIES, INK, MUTED, GRID, SURFACE = "#2a78d6", "#52514e", "#898781", "#e9e8e4", "#fcfcfb"
SEED = 7

TITLE = "Vulnerable customer capacity planner"
SUBTITLE = "How many specialist handoffs does a catch-rate target need?"

st.set_page_config(page_title=TITLE, layout="wide")


@st.cache_data(show_spinner=False)
def demo_data(assumptions_json: str) -> pd.DataFrame:
    return generate(Assumptions.from_json(assumptions_json), seed=SEED)


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def per_1000(x: float) -> str:
    """A rate per 1,000 without a trailing '.0': 60 or 59.5."""
    return f"{x:.1f}".removesuffix(".0")


def threshold_text(t: float | None) -> str:
    return "above all scores" if t is None else f"{t:.3f}"


st.title(TITLE)
st.markdown(f"#### {SUBTITLE}")
st.warning(PITCH["banner"]["text"].format(firm=FIRM))

# ---------------------------------------------------------------------------------------------------------------
# Sidebar: data source, capacity, and (for the demo) every assumption, each tagged "assumed"
# ---------------------------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Data")
    mode = st.radio("Data source", ["Synthetic demo data", "Upload a CSV"], key="mode")
    source = "demo" if mode == "Synthetic demo data" else "upload"
    st.download_button(
        "Download the sample CSV (synthetic)",
        (ROOT / "data" / "sample_synthetic.csv").read_bytes(),
        file_name="sample_synthetic.csv",
        mime="text/csv",
    )
    with st.expander("CSV schema"):
        st.markdown(
            "One row per conversation:\n"
            "- `conversation_id`: unique text\n"
            "- `score`: the guardrail score, 0 to 1\n"
            "- `warrants_specialist`: 1 if the conversation warrants a specialist route, else 0\n"
            f"- `characteristic`: one of {', '.join(f'`{c}`' for c in ALLOWED_CHARACTERISTICS)} "
            "when `warrants_specialist` is 1; blank when it is 0\n\n"
            "Labels should come from a QA-reviewed sample, stratified in proportion to traffic, that includes "
            "conversations the guardrail did not hand off. Misses are invisible otherwise. Every row counts "
            "equally: there are no weights."
        )

    st.header("Capacity")
    staffed = st.number_input(
        label("Staffed specialist capacity, handoffs per 1,000", ASSUMED),
        min_value=0.0, max_value=1000.0, value=150.0, step=5.0, key="staffed",
    )

    if source == "demo":
        st.header("Assumptions")
        st.caption("Every value in this section is assumed, not measured. Edit it to match your traffic.")
        prevalence = st.slider(
            label("Share that warrants a specialist route (%)", ASSUMED), 1, 20, 6, key="prevalence"
        )
        st.markdown(f"**Characteristic mix** · {ASSUMED} (weights, normalised)")
        mix = {
            c: st.number_input(CHARACTERISTIC_LABELS[c], 0, 100, 25, 5, key=f"mix_{c}")
            for c in CHARACTERISTICS
        }
        st.markdown(f"**Guardrail score for conversations that warrant a specialist** · {ASSUMED}")
        positive_scores = {}
        for c in CHARACTERISTICS:
            a, b = st.columns(2)
            mean = a.slider(f"{CHARACTERISTIC_LABELS[c]}: mean", 0.05, 0.95, 0.70, 0.05, key=f"mean_{c}")
            spread = b.selectbox("spread", list(SPREADS), index=1, key=f"spread_{c}")
            positive_scores[c] = ScoreDist(mean, spread)
        st.markdown(f"**Conversations that do not** · {ASSUMED}")
        hard_share = st.slider("Hard negatives: ordinary hardship and complaints (%)", 0, 100, 25, 5, key="hard_share")
        a, b = st.columns(2)
        hard_mean = a.slider("Hard negatives: mean", 0.05, 0.95, 0.45, 0.05, key="hard_mean")
        hard_spread = b.selectbox("spread", list(SPREADS), index=1, key="hard_spread")
        a, b = st.columns(2)
        routine_mean = a.slider("Routine: mean", 0.05, 0.95, 0.10, 0.05, key="routine_mean")
        routine_spread = b.selectbox("spread", list(SPREADS), index=1, key="routine_spread")
        st.caption(f"N = 20,000 synthetic conversations, seed {SEED}, reported per 1,000 · {ASSUMED}")

# ---------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------
if source == "demo":
    try:
        assumptions = Assumptions(
            prevalence=prevalence / 100,
            characteristic_mix=mix,
            positive_scores=positive_scores,
            hard_negative_share=hard_share / 100,
            hard_negative_scores=ScoreDist(hard_mean, hard_spread),
            routine_scores=ScoreDist(routine_mean, routine_spread),
        )
        df = demo_data(assumptions.to_json())
    except ValueError as e:
        st.error(f"Those assumptions can't generate data: {e}")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV", type="csv", key="upload")
    if uploaded is None:
        st.info("Upload a CSV in the sidebar to continue, or switch back to the synthetic demo data.")
        st.stop()
    try:
        raw = pd.read_csv(uploaded, dtype=str, keep_default_na=False)
    except Exception as e:  # a malformed file should explain itself, not crash the app
        st.error(f"That file could not be read as a CSV: {e}")
        st.stop()
    df, errors = validate_upload(raw)
    if errors:
        st.error("The file doesn't match the schema:\n\n" + "\n".join(f"- {m}" for m in errors))
        st.stop()

r = result_basis(source)
scores, labels = df["score"].to_numpy(), df["warrants_specialist"].to_numpy()
n = len(df)
footnote = footnote_for(source, n, SEED)
if source == "upload":
    # Every result below rests on the same labels, so the warning sits once above all of them.
    st.caption(SAMPLING_WARNING)

# ---------------------------------------------------------------------------------------------------------------
# Panel 1 (leads): target catch rate -> the capacity it needs, flagged against staffed capacity
# ---------------------------------------------------------------------------------------------------------------
st.subheader("1. Set a target catch rate: what specialist capacity does it need?")
target = st.slider(label("Target catch rate (%)", ASSUMED), 50, 100, 90, key="target") / 100
need = cal.capacity_for_target_catch_rate(scores, labels, target)
short = cal.shortfall_per_1000(need.handoffs_per_1000, staffed)

c1, c2, c3 = st.columns(3)
c1.metric(label("Required capacity, handoffs per 1,000", r), f"{need.handoffs_per_1000:.1f}")
c2.metric(label("Staffed capacity, handoffs per 1,000", ASSUMED), f"{staffed:.1f}")
c3.metric(label("Threshold: hand off when score ≥", r), threshold_text(need.threshold))
if short > 0:
    st.error(
        f"**Shortfall: {short:.1f} handoffs per 1,000** ({r}). A {target:.0%} catch rate needs "
        f"{need.handoffs_per_1000:.1f} per 1,000, but staffed capacity is {staffed:.1f} ({ASSUMED})."
    )
else:
    st.success(
        f"**Within capacity** ({r}). A {target:.0%} catch rate needs {need.handoffs_per_1000:.1f} per 1,000, "
        f"leaving {staffed - need.handoffs_per_1000:.1f} of the {staffed:.1f} staffed ({ASSUMED})."
    )
st.caption(f"At this threshold: precision {pct(need.precision)}, catch rate {pct(need.catch_rate)} · {r}")

# ---------------------------------------------------------------------------------------------------------------
# Panel 2: capacity -> catch rate and precision
# ---------------------------------------------------------------------------------------------------------------
st.subheader("2. Or set a capacity: what does it catch?")
capacity = st.slider(
    label("What if capacity were… (handoffs per 1,000)", ASSUMED), 0, 300, 150, 1, key="capacity"
)
capacity_count = cal.per_1000_to_count(capacity, n)
point = cal.operating_point_for_capacity(scores, labels, capacity_count)
d1, d2, d3 = st.columns(3)
d1.metric(label("Catch rate", r), pct(point.catch_rate))
d2.metric(label("Precision", r), pct(point.precision))
d3.metric(label("Threshold: hand off when score ≥", r), threshold_text(point.threshold))
st.caption(
    f"Uses {point.handoffs_per_1000:.1f} of {capacity} handoffs per 1,000: the lowest threshold that stays "
    "within capacity. Tied scores are handed off together, so a tied group that would overshoot stays out."
)

# The hard ceiling: what even a perfect guardrail could do at this capacity. Compared in whole conversations.
positives = int(labels.sum())
pos_per_1000 = 1000.0 * positives / n
if capacity_count >= positives:
    st.caption(
        f"**No hard ceiling at {capacity} per 1,000** ({ASSUMED}): a perfect guardrail could catch all "
        f"{per_1000(pos_per_1000)} per 1,000 that warrant a specialist ({r})."
    )
else:
    perfect_count = cal.perfect_guardrail_need(labels, target)
    ceiling = (
        f"**Hard ceiling at {capacity} per 1,000** ({ASSUMED}): even a perfect guardrail catches at most "
        f"{pct(cal.max_catch_rate_at_capacity(labels, capacity_count))} "
        f"({per_1000(1000.0 * capacity_count / n)} of {per_1000(pos_per_1000)} per 1,000 that warrant a "
        f"specialist; {r}). A {target:.0%} target ({ASSUMED}) needs at least {1000.0 * perfect_count / n:.1f} "
        f"per 1,000 however good the guardrail ({r})."
    )
    if capacity_count < perfect_count:
        st.error(ceiling)
    else:
        st.caption(ceiling)

# ---------------------------------------------------------------------------------------------------------------
# Chart: catch rate against capacity, one series, with target and staffed capacity as reference lines
# ---------------------------------------------------------------------------------------------------------------
cv = cal.curve(scores, labels)
x_max = float(max(300.0, staffed, need.handoffs_per_1000) * 1.1)
cv = cv[cv["handoffs_per_1000"] <= x_max]
if len(cv) > 600:  # thin the line for the browser; the arithmetic above uses every row
    cv = cv.iloc[np.unique(np.linspace(0, len(cv) - 1, 600).astype(int))]

x = alt.X("handoffs_per_1000:Q", title="Specialist handoffs per 1,000 conversations", scale=alt.Scale(domain=[0, x_max]))
y = alt.Y("catch_rate:Q", title="Catch rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]))
hover = alt.selection_point(nearest=True, on="pointerover", fields=["handoffs_per_1000"], empty=False)
base = alt.Chart(cv).encode(x=x, y=y)
line = base.mark_line(color=SERIES, strokeWidth=2)
points = base.mark_point(color=SERIES, size=80, filled=True).encode(
    opacity=alt.condition(hover, alt.value(1), alt.value(0)),
    tooltip=[
        alt.Tooltip("handoffs_per_1000:Q", title="Handoffs per 1,000", format=".1f"),
        alt.Tooltip("catch_rate:Q", title="Catch rate", format=".1%"),
        alt.Tooltip("precision:Q", title="Precision", format=".1%"),
        alt.Tooltip("threshold:Q", title="Threshold", format=".3f"),
    ],
).add_params(hover)
crosshair = alt.Chart(cv).mark_rule(color=MUTED, strokeWidth=1).encode(x="handoffs_per_1000:Q").transform_filter(hover)
refs = pd.DataFrame({"staffed": [staffed], "target": [target]})
staffed_rule = alt.Chart(refs).mark_rule(color=INK, strokeDash=[4, 4]).encode(x="staffed:Q")
target_rule = alt.Chart(refs).mark_rule(color=INK, strokeDash=[4, 4]).encode(y="target:Q")
staffed_text = alt.Chart(refs).mark_text(align="left", dx=4, dy=-6, color=INK, baseline="bottom").encode(
    x="staffed:Q", y=alt.value(14), text=alt.value(f"Staffed capacity ({ASSUMED})")
)
target_text = alt.Chart(refs).mark_text(align="right", dx=-4, dy=-6, color=INK).encode(
    x=alt.datum(x_max), y="target:Q", text=alt.value(f"Target ({ASSUMED})")
)
chart = (
    (line + points + crosshair + staffed_rule + target_rule + staffed_text + target_text)
    .properties(height=320, title=label("Catch rate by specialist capacity", r), background=SURFACE)
    .configure_axis(gridColor=GRID, domainColor=MUTED, tickColor=MUTED, labelColor=INK, titleColor=INK)
    .configure_title(color=INK, anchor="start")
    .configure_view(strokeWidth=0)
)
st.altair_chart(chart, width="stretch")
st.caption(f"Dashed lines: target catch rate {target:.0%} and staffed capacity {staffed:.1f} per 1,000 ({ASSUMED}).")
with st.expander("The curve as a table"):
    rows = []
    for c in range(10, 301, 10):
        p = cal.operating_point_for_capacity(scores, labels, cal.per_1000_to_count(c, n))
        rows.append({"Capacity per 1,000": c, "Catch rate": pct(p.catch_rate), "Precision": pct(p.precision)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------------------------------------------
# Misses by characteristic, at the panel 1 threshold
# ---------------------------------------------------------------------------------------------------------------
st.subheader(label("Missed conversations by FG21/1 characteristic, at the panel 1 threshold", r))
misses = cal.misses_by_characteristic(df, need.threshold)
st.dataframe(
    pd.DataFrame(
        {
            "Characteristic": [CHARACTERISTIC_LABELS.get(c, c) for c in misses["characteristic"]],
            "Warrant a specialist": misses["positives"],
            "Missed": misses["missed"],
            "Missed share": [pct(v) for v in misses["missed_share"]],
        }
    ),
    hide_index=True,
    width="stretch",
)
noise = (
    f"With about {int(misses['positives'].median())} conversations per characteristic, differences of a few "
    "points between rows can be sampling noise."
)
if source == "demo":
    st.caption(
        "This pattern follows from the assumptions above; it is not a finding about real traffic. "
        + noise + " Here the assumptions are what differ, or nothing does."
    )
else:
    st.caption("Computed from your data. " + noise)

# ---------------------------------------------------------------------------------------------------------------
# MI export (one markdown table), About, credits
# ---------------------------------------------------------------------------------------------------------------
st.subheader("MI export")
md = mi_table_markdown(mi_rows(target, need, staffed, short, misses, source), footnote)
st.download_button("Download the MI table (markdown)", md, file_name="mi_table.md", mime="text/markdown")
with st.expander("Preview the MI table"):
    st.markdown(md)

with st.expander("About this planner"):
    st.markdown(PITCH["about"]["text"].format(firm=FIRM))
    st.markdown("**Sources**")
    st.markdown("\n".join(f"- [{s['title']}]({s['url']}), {s['author']}, {s['date']}" for s in PITCH["sources"]))

st.caption("Built by Manan Ahuja with Claude Code.")
