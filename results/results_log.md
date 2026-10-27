# Results Log

Append one row every time you run an experiment, even if the result is
bad or preliminary. This table becomes the source for your report's
results section — don't let it go stale.

| Date | Representation | Model | PR-AUC | F1 | Utility Score | Notes | Run by |
|---|---|---|---|---|---|---|---|
| | A | Logistic Regression | | | | | |
| | A | HistGradientBoosting | | | | | |
| | B | Logistic Regression | | | | | |
| | B | HistGradientBoosting | | | | | |
| | C | Logistic Regression | | | | | |
| | C | HistGradientBoosting | | | | | |

## How to fill this in

- **Representation**: A (imputation only) / B (+ missingness masks) / C (+ time-since-measured)
- **Utility Score**: from PhysioNet's official `evaluate_sepsis_score.py`, not a custom implementation
- **Notes**: anything that affects interpretation — e.g. "threshold=0.5, not tuned yet", "ran on val set only"
- Always report **val set** results here until the final paper, where test set results appear once, at the end
