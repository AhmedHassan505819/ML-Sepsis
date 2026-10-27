# Early Sepsis Prediction from ICU Time Series
### Does modelling missingness patterns improve prediction?

ML course semester project — see `reports/synopsis.pdf` for the full proposal.

## Team
- Name 1 — CMS ID
- Name 2 — CMS ID
- Name 3 — CMS ID

## Project summary

We compare three data representations on the PhysioNet/CinC 2019 Sepsis
Challenge dataset to test whether modelling *missingness patterns* (which
lab was measured when, not just its value) improves early sepsis prediction:

- **Representation A** — standard imputation only (baseline)
- **Representation B** — imputation + binary "was this measured?" masks
- **Representation C** — imputation + time-since-last-measurement features

Same models (Logistic Regression, Gradient Boosting), same patient-level
train/val/test split, evaluated with PR-AUC, F1, and the Challenge's
official time-dependent utility score.

## Repo structure

```
data/         How to obtain the data (raw data is NOT committed - too large for git)
notebooks/    Numbered notebooks, one per pipeline stage (01, 02, 03...)
src/          Reusable functions imported by the notebooks/scripts
scripts/      Standalone runnable .py pipeline scripts
results/      Metrics log + figures generated during experiments
reports/      LaTeX source + PDFs for synopsis / mid-sem report / final paper
docs/         Literature notes, meeting notes, misc planning docs
```

## How to reproduce

1. `pip install -r requirements.txt`
2. Get the data — see `data/README.md`
3. Run `scripts/step1_split_and_baseline.py` to generate the patient-level
   split and Representation A baseline results
4. Work through `notebooks/` in numbered order for the full pipeline

## Workflow rules (for the three of us)

- Don't push directly to `main`. Branch per task (`yourname/task-name`),
  open a PR, merge when it runs cleanly.
- Clear notebook outputs before committing (`Cell → All Output → Clear`)
  so diffs stay readable.
- Keep real logic in `src/*.py`, keep notebooks thin (import + call +
  plot). Avoids merge conflicts on the same notebook.
- Every experiment you run gets one row in `results/results_log.md`,
  even if the result is bad — that's still evidence for the report.
