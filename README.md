# Vi_home_assignment

## Quick Start
```bash
# create & activate virtualenv
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# generate features
.venv/bin/python src/feature_plan.py --data-dir data/train --output-dir feature_outputs
.venv/bin/python src/feature_plan.py --data-dir data/test --output-dir feature_outputs_test

# train both tasks (churn + benefit)
.venv/bin/python src/train_models.py --model logistic_regression --task churn --features-path feature_outputs/features.csv --output-dir model_outputs/churn
.venv/bin/python src/train_models.py --model logistic_regression --task benefit --features-path feature_outputs/features.csv --output-dir model_outputs/benefit

# evaluate on holdout
.venv/bin/python src/evaluate_model.py --model-path model_outputs/churn/logistic_regression.joblib --features-path feature_outputs_test/features.csv --output-dir model_outputs --split-name churn_test --task churn
.venv/bin/python src/evaluate_model.py --model-path model_outputs/benefit/logistic_regression.joblib --features-path feature_outputs_test/features.csv --output-dir model_outputs --split-name benefit_test --task benefit

# produce ranked outreach lists (top N)
.venv/bin/python src/recommend_outreach.py --model-path model_outputs/churn/logistic_regression.joblib --features-path feature_outputs/features.csv --output-dir recommendations_churn --top-n 500 --task churn
.venv/bin/python src/recommend_outreach.py --model-path model_outputs/benefit/logistic_regression.joblib --features-path feature_outputs/features.csv --output-dir recommendations_benefit --top-n 500 --task benefit
```

## Approach
- **Data understanding:** ran `src/run_eda.py` against the `data/train` tables to profile member coverage, engagement volume, ICD mix, and churn/outreach rates during the 14-day observation window.
- **Key EDA takeaways:** members average ~10 app sessions, ~26 web visits, and ~6.5 claims, with ICD-10 codes Z71.3, I10, and E11.9 dominating; engagement coverage is nearly complete across channels.
- **Outreach impact insight:** outreach members show a ~1.3 pp lower churn rate (19.4 % vs 20.7 %), but binomial confidence intervals overlap, so this difference is likely statistical noise unless further tests or covariate adjustments show otherwise.

## Feature Engineering
- **Feature set (`src/feature_plan.py`):**  
  - Engagement volume: total app sessions, web visits, claims, and their daily averages to capture intensity.  
  - Recency & cadence: days since the last app/web/claim interaction plus the std-dev of inter-event intervals to reflect freshness and regularity.  
  - Clinical markers: binary flags for priority ICD-10 codes (I10, E11.9, Z71.3) and an aggregate `chronic_any` indicator aligned with WellCo’s cohorts.  
  - Ratios & treatment interactions: app↔web mix, claim↔app ratio, outreach flag, low-engagement flag, and an outreach×low-engagement interaction to surface heterogeneous treatment effects.
- **Refinement:** removed `claims_icd_diversity` because it was nearly perfectly correlated with `claims_total`, preventing redundant signals from dominating importance or correlation analyses.
- **Usage:**  
  ```bash
  # Train split
  MPLCONFIGDIR=.matplotlib .venv/bin/python src/feature_plan.py \
    --data-dir data/train \
    --output-dir feature_outputs

  # Test split (files prefixed with test_)
  MPLCONFIGDIR=.matplotlib .venv/bin/python src/feature_plan.py \
    --data-dir data/test \
    --output-dir feature_outputs_test
  ```

## Modeling
We maintain two logistic-regression experiments, selected via `--task`:

1. `--task churn` – predicts churn probability in the post-outreach window (outreach features included).
2. `--task benefit` – drops outreach features and labels positives as `(treated & stayed) OR (untreated & churned)` to identify members who benefit from outreach.

**Training commands**
```bash
# Churn
.venv/bin/python src/train_models.py \
  --model logistic_regression \
  --task churn \
  --features-path feature_outputs/features.csv \
  --output-dir model_outputs/churn

# Benefit
.venv/bin/python src/train_models.py \
  --model logistic_regression \
  --task benefit \
  --features-path feature_outputs/features.csv \
  --output-dir model_outputs/benefit
```
Each run writes `logistic_regression.joblib`, `logistic_regression_metrics.json`, and `logistic_regression_report.txt` under the respective output directory.

## Evaluation
- **Evaluation script (`src/evaluate_model.py`):** scores a trained pipeline plus any feature CSV (train/test) and writes `<split>_metrics.json` + `_report.txt`.
- **Usage:**  
  ```bash
  # Churn task on test split
  .venv/bin/python src/evaluate_model.py \
    --model-path model_outputs/churn/logistic_regression.joblib \
    --features-path feature_outputs_test/features.csv \
    --output-dir model_outputs \
    --split-name churn_test \
    --task churn

  # Benefit task on test split
  .venv/bin/python src/evaluate_model.py \
    --model-path model_outputs/benefit/logistic_regression.joblib \
    --features-path feature_outputs_test/features.csv \
    --output-dir model_outputs \
    --split-name benefit_test \
    --task benefit
  ```
- Metrics include accuracy, precision, recall, F1, ROC AUC, and the confusion matrix stored in JSON for downstream reporting.

## Model Interpretability
- **Churn model coefficients:**  
  ```bash
  MPLCONFIGDIR=.matplotlib .venv/bin/python interpretability/logistic_regression_analysis.py \
    --model-path model_outputs/churn/logistic_regression.joblib \
    --features-path feature_outputs/features.csv \
    --output-dir interpretability_outputs/churn \
    --top-k 15
  ```
  Shows which features raise or lower churn risk (e.g., chronic ICDs + high app engagement reduce churn; erratic web behavior lifts it).

- **Benefit model coefficients:**  
  ```bash
  MPLCONFIGDIR=.matplotlib .venv/bin/python interpretability/logistic_regression_analysis.py \
    --model-path model_outputs/benefit/logistic_regression.joblib \
    --features-path feature_outputs/features.csv \
    --output-dir interpretability_outputs/benefit \
    --top-k 15
  ```
  Highlights which pre-outreach signals indicate someone benefited from outreach or would have if contacted (e.g., low engagement, channel ratios).

## Outreach Prioritization
- **Ranking script (`src/recommend_outreach.py`):** sorts members by model score and writes `ranked_scores.csv`, `top_candidates.csv`, `summary.json`, and `lift_plot.png`. Use the matching `--task`.
  ```bash
  # Churn list (post-outreach risk)
  .venv/bin/python src/recommend_outreach.py \
    --model-path model_outputs/churn/logistic_regression.joblib \
    --features-path feature_outputs/features.csv \
    --output-dir recommendations_churn \
    --top-n 500 \
    --evaluation-counts 100,200,500,1000,2000,3000,5000 \
    --plot-counts 100,200,500,1000,2000,3000,4000,5000 \
    --task churn

  # Benefit list (people to outreach next)
  .venv/bin/python src/recommend_outreach.py \
    --model-path model_outputs/benefit/logistic_regression.joblib \
    --features-path feature_outputs/features.csv \
    --output-dir recommendations_benefit \
    --top-n 500 \
    --evaluation-counts 100,200,500,1000,2000,3000,5000 \
    --plot-counts 100,200,500,1000,2000,3000,4000,5000 \
    --task benefit
  ```
  - Churn output highlights high-risk members regardless of treatment (use to monitor overall churn lift).
  - Benefit output emphasizes members who either benefited from outreach or would benefit if contacted. Filter `ranked_scores.csv` to `outreach == 0` to build the next outreach wave, while `outreach == 1` entries describe traits of outreach successes.
