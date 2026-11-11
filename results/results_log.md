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

 | A | Logistic Regression | 0.0686 | 0.0775 | | val set, thr=0.5, balanced weights, single run | 
 | A | HistGradientBoosting | 0.0886 | 0.1120 | | val set, thr=0.5, balanced weights, default params, seed 42 |

 ## Split (notebook 02, seed 42, stratified by ever-septic)
| Split | Patients | Rows | Ever-septic patients |
|---|---|---|---|
| train | 28,235 | 1,086,436 | 7.27% |
| val   | 6,050  | 231,472   | 7.27% |
| test  | 6,051  | 234,302   | 7.27% |

Random-guess PR-AUC on val = 0.018 (4,162 positive rows / 231,472).
Test split is untouched until the final paper.


| A | HistGradientBoosting | 0.0869 | 0.1124 |  | val set, thr=0.5, mean of 3 run(s) | |
| B | HistGradientBoosting | 0.0900 | 0.1150 |  | val set, thr=0.5, mean of 3 run(s) | |
| A | LogisticRegression | 0.0686 | 0.0775 |  | val set, thr=0.5, mean of 1 run(s) | |
| B | LogisticRegression | 0.0750 | 0.0862 |  | val set, thr=0.5, mean of 1 run(s) | |


| A | HistGradientBoosting | 0.0869 | 0.1124 |  | val set, thr=0.5, mean of 3 run(s) | |
| B | HistGradientBoosting | 0.0900 | 0.1150 |  | val set, thr=0.5, mean of 3 run(s) | |
| C | HistGradientBoosting | 0.0933 | 0.1180 |  | val set, thr=0.5, mean of 3 run(s) | |
| A | LogisticRegression | 0.0686 | 0.0775 |  | val set, thr=0.5, mean of 1 run(s) | |
| B | LogisticRegression | 0.0750 | 0.0862 |  | val set, thr=0.5, mean of 1 run(s) | |
| C | LogisticRegression | 0.0844 | 0.0969 |  | val set, thr=0.5, mean of 1 run(s) | |