# Vulnerable customer capacity planner

**How many specialist handoffs does a catch-rate target need?**

> **Unofficial prototype.** Built from public information only. It is not affiliated with or endorsed by any firm
> named here, and it uses no internal data. The demo runs on **synthetic data**: every result follows from stated
> assumptions and is not a finding about real traffic.

A guardrail that spots customers in vulnerable circumstances hands some conversations to a specialist team. Where
you set its threshold decides two things at once: how many of those customers reach a specialist (the catch
rate), and how many handoffs the team has to absorb. This planner shows the trade-off in both directions.

1. **Set a target catch rate, see the specialist capacity it needs.** The shortfall is flagged red when the need
   exceeds the capacity you have staffed.
2. **Set a capacity, see what it catches**, with the precision at that capacity.

## The problem

The FCA's guidance says "Firms should be able to spot signs of vulnerability"
([FG21/1](https://www.fca.org.uk/publications/finalised-guidance/guidance-firms-fair-treatment-vulnerable-customers),
23 Feb 2021, updated 22 Jul 2026). Its 2025 review found that "most firms were unable to effectively monitor
outcomes for customers in vulnerable circumstances", and that identifying signs of vulnerability "was challenging
for most firms. In particular, we saw this in firms that have primarily digital customer journeys"
([FCA, 7 Mar 2025](https://www.fca.org.uk/publications/good-and-poor-practice/delivering-vulnerable-customers)).
In its September 2026 review of a sample of payments firms, the FCA reported that "In some cases" there "was
limited testing or assurance to assess how effectively policies were being implemented in practice"
([FCA, 17 Sep 2026](https://www.fca.org.uk/publications/good-and-poor-practice/payments-firms-delivering-good-outcomes-vulnerable-consumers)).

Automated channels increasingly do the first spotting. Their thresholds are usually tuned for precision and catch
rate. This planner adds the staffing side: what a catch-rate target costs in specialist time, and what a given
team size can actually catch.

## The approach

- **One row per conversation.** Each row holds a guardrail `score` (0 to 1) and a label, `warrants_specialist`:
  *a conversation that warrants a specialist route*. A positive row also records which FG21/1 characteristic of
  vulnerability applies: health, life events, resilience or capability.
- **The threshold rule.** A conversation is handed off when `score ≥ threshold`. The candidate thresholds are the
  observed scores, so every result is one a real threshold can produce: tied scores go together or not at all.
- **Target → capacity.** The planner picks the **highest** threshold that still reaches the target, so it uses
  the fewest handoffs. The capacity it needs is compared with staffed capacity.
- **Capacity → catch rate.** The planner picks the **lowest** threshold whose handoffs stay within capacity, so it
  uses as much of the capacity as possible. A tied group that would overshoot stays out.
- **The idea behind it** comes from anti-money-laundering triage: when a team can review only so many cases,
  judge the system by precision at that fixed capacity. For vulnerable customers, capacity is also something to
  plan *for*, not only a budget to live within. That is why the target-first view leads.
- **Tags.** Every number on screen is tagged by its basis:
  - **assumed:** an input
  - **synthetic:** a result on the demo data
  - **from your data:** a result on an uploaded file

## How to run it

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
.venv/bin/pytest -q          # 80 tests
```

To regenerate the sample file: `.venv/bin/python -m src.synth --write-sample data/sample_synthetic.csv`.

## Two-minute demo path

1. **Open the app.** The target is 90% and the screen is green: 132.5 handoffs per 1,000 are needed against 150
   staffed.
2. **Raise the target to 95%.** It turns red: 177.1 are needed, a shortfall of 27.1. Five more points of catch
   rate cost 44.6 more handoffs per 1,000.
3. **Drag "What if capacity were…" down to 50.** The hard-ceiling message appears: even a perfect guardrail
   catches at most 83.3% (50 of 60), and a 95% target needs at least 57.0 per 1,000 however good the guardrail.

All figures are from the synthetic demo data and its stated assumptions.

## Using your own data

Upload a CSV with one row per conversation. You can download a sample from the app.

| Column | Meaning |
|---|---|
| `conversation_id` | Unique text |
| `score` | The guardrail score, from 0 to 1 |
| `warrants_specialist` | 1 if the conversation warrants a specialist route, otherwise 0 |
| `characteristic` | `health`, `life_events`, `resilience`, `capability` or `unknown` when `warrants_specialist` is 1; blank when it is 0 |

Faults are reported by row number. Rows are counted from 1, excluding the header.

### Where real labels come from

On real traffic, `warrants_specialist` has to come from a **QA-reviewed sample**. That sample must include
conversations the guardrail did *not* hand off, because misses are invisible otherwise. A sample drawn only from
handoffs can measure precision, but never catch rate.

- **Sample** at random, stratified by score band **in proportion to traffic** and including low scores, so that
  rows need no weights. This version counts every uploaded row equally, so a sample that over-samples some score
  bands would bias precision and catch rate.
- **Label** each conversation against the firm's vulnerability policy, recording the FG21/1 characteristic where
  one applies.
- **Spot-check** agreement between reviewers.

That is the difference between a planning demo on synthetic data and assurance on a firm's own labels.

## Limitations

- **On the demo data, every number follows from the assumptions.** Each assumption is tagged and editable, and
  none is a measurement.
- **Small groups are noisy.** With a few hundred conversations per characteristic, differences of a few points
  between rows can be sampling noise. There are no confidence intervals yet.
- **Labels are the main risk.** Results are only as good as the labels, and the planner trusts them. It checks
  that a file is well-formed, not that the sample was drawn properly: a sample of handed-off conversations alone
  would hide every miss and overstate the catch rate. Whenever a file is uploaded, the app says so above the
  results and in the MI export's footnote.
- **One guardrail, one route.** There is no three-way routing, for example continuing with a hardship procedure
  instead of handing off.
- **Steady traffic is assumed.** Capacity per 1,000 conversations says nothing about peaks or time of day.
- **No weights.** Uploaded rows are unweighted (see above).
- **It doesn't choose the target.** What catch rate is right is a policy decision for the firm.

## Decision log

| Decision | Instead of | Why |
|---|---|---|
| Assumed score distributions, no classifier | Training a text classifier on synthetic messages | The tool is about the threshold decision. A classifier trained and scored on template text would be graded by the method that generated its data. A classifier comes later, and only on data it wasn't generated from. |
| Target-first panel leads | Leading with precision and catch rate at a threshold | Precision and catch rate are already standard guardrail metrics. Capacity is the missing view. |
| Capacity: lowest threshold within capacity | Highest threshold that meets the constraint | Read literally, the highest one gives zero handoffs. The lowest one uses the capacity. A test checks that one step lower would overshoot. |
| Target: highest threshold that reaches it | Any threshold that reaches it | Fewest handoffs for the same catch rate. |
| Thresholds are the observed scores | A fixed 0.01 grid | Exact, and every operating point is achievable. Ties are handled honestly. |
| Exact counts (largest remainder), random scores only | Sampling the counts too | Prevalence and mix are assumptions, so they should hold exactly. Only the scores vary. |
| Beta distributions set by mean and spread | Asking for alpha and beta | Bounded to 0 to 1, and operators think in "average score", not shape parameters. |
| Equal default assumptions across characteristics | Different defaults | No pattern is built in. The demo changes one assumption to show sensitivity, and a caption explains sampling noise. |
| One helper for the basis tags | Tagging in each panel | The screen and the MI export can't disagree. |
| No paid API | An LLM to label or explain | Nothing at runtime needs a model, so a public link can't run up a bill. |
| Firm-specific text only in `pitch.toml` | Firm names in the code | The tool can be reused for other firms, and a test checks the code names none. |
| Banner says "Synthetic demo data unless you upload your own" | "Synthetic data." | After an upload, the data isn't synthetic. |
| Defaults: 6% warrant a specialist, 150 staffed per 1,000; the what-if slider starts at 150 | 50 staffed, which opens red | Changed 11 Nov 2026. At 50, even a perfect guardrail can't reach 90%, so the screen opened red. At 150 the demo opens green at 90%, turns red at 95%, and the what-if slider shows the hard ceiling. |
| Target capacity counted in integer maths on whole percents | `math.ceil(target * positives)` | Float error can add one: `0.56 * 100` is `56.00000000000001`. Tested: 56% of 100 positives needs 56, not 57. |
| A hard-ceiling line under panel 2 | Leave it implicit | However good the guardrail, a capacity below the number of conversations that warrant a specialist caps the catch rate. That is a staffing fact, not a model fact, so it is stated. |
| Capacity per 1,000 rounds down | Rounding to nearest | Never exceed the capacity entered. |

## Data sources

- `data/sample_synthetic.csv`: 2,000 synthetic rows from the default assumptions, seed 7. See `data/SOURCES.md`.
- The regulatory sources are cited in "The problem" above, with dates. All were accessed on 10 Nov 2026.

## Context for this pitch

*This is the only firm-specific section. To reuse the planner for another firm, rewrite it and edit `pitch.toml`.*

Prepared as an unofficial prototype for **Gradient Labs**. Their guide on vulnerable customers advises: "Calibrate
the threshold against your own traffic rather than accepting a vendor default". It lists the per-guardrail numbers
they log: "run count, hand-off count, precision, catch rate, and accuracy". Source:
[FCA vulnerable customers: how AI agents detect and escalate](https://gradient-labs.ai/guides/fca-vulnerable-customers-how-ai-agents-detect-and-escalate),
Elizabeth Shew, 2 Nov 2026. This planner looks at the same threshold from the staffing side.

None of Gradient Labs' data, outputs or systems were used, simulated or tested.

---

Built by Manan Ahuja with Claude Code.
