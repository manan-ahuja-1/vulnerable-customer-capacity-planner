# Data sources

## `sample_synthetic.csv`
Synthetic. It was generated on 10 Nov 2026 with:

```
python -m src.synth --write-sample data/sample_synthetic.csv   # 2,000 rows, seed 7, default assumptions
```

The default assumptions are in `src/synth.py` (`Assumptions`):
- **Size and prevalence.** 6% of conversations warrant a specialist route.
- **Mix.** The four FG21/1 characteristics are weighted equally.
- **Scores for positives.** Beta-distributed, mean 0.70 with medium spread, for every characteristic.
- **Negatives.** 25% of them are "hard negatives" (ordinary hardship and complaints) with mean 0.45; the rest are
  routine conversations with mean 0.10.

None of these is a measurement. No real conversations, customers or firm data were used.

## Regulatory and public sources (accessed 10 Nov 2026)
- FCA, [FG21/1: Guidance for firms on the fair treatment of vulnerable customers](https://www.fca.org.uk/publications/finalised-guidance/guidance-firms-fair-treatment-vulnerable-customers), 23 Feb 2021, updated 22 Jul 2026
- FCA, [Delivering good outcomes for customers in vulnerable circumstances](https://www.fca.org.uk/publications/good-and-poor-practice/delivering-vulnerable-customers), 7 Mar 2025
- FCA, [Payments firms: delivering good outcomes for consumers in vulnerable circumstances](https://www.fca.org.uk/publications/good-and-poor-practice/payments-firms-delivering-good-outcomes-vulnerable-consumers), 17 Sep 2026
- The firm-specific source is listed in `pitch.toml` and the README's "Context for this pitch".
