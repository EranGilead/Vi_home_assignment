# Vi_home_assignment

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
- **Training script (`src/train_models.py`):** loads `feature_outputs/features.csv`, drops redundant columns (e.g., `claims_icd_diversity`), splits the data, scales numeric inputs when needed, and fits one of three models: logistic regression (class-weighted), random forest, or XGBoost.
- **Commands:**  
  ```bash
  # Logistic regression
  .venv/bin/python src/train_models.py \
    --model logistic_regression \
    --features-path feature_outputs/features.csv \
    --output-dir model_outputs

  # Random forest
  .venv/bin/python src/train_models.py \
    --model random_forest \
    --features-path feature_outputs/features.csv \
    --output-dir model_outputs \
    --rf-n-estimators 500 \
    --rf-max-depth 10

  # XGBoost
  .venv/bin/python src/train_models.py \
    --model xgboost \
    --features-path feature_outputs/features.csv \
    --output-dir model_outputs \
    --xgb-n-estimators 800 \
    --xgb-learning-rate 0.05 \
    --xgb-max-depth 4 \
    --xgb-subsample 0.8 \
    --xgb-colsample 0.8
  ```
- **Artifacts:** each run saves a `<model>.joblib` plus `<model>_metrics.json`/`<model>_report.txt` describing accuracy, precision/recall/F1, ROC AUC, and confusion matrix.

## Evaluation
- **Evaluation script (`src/evaluate_model.py`):** loads any trained pipeline plus a feature CSV (with labels) and writes `<split>_metrics.json` and `<split>_report.txt` under `model_outputs/`.
- **Usage:**  
  ```bash
  .venv/bin/python src/evaluate_model.py \
    --model-path model_outputs/logistic_regression.joblib \
    --features-path feature_outputs_test/features.csv \
    --output-dir model_outputs \
    --split-name test
  ```
- **Results:** evaluated with `src/evaluate_model.py`  
  - Logistic regression (test): accuracy ≈0.544, churn recall ≈0.543, F1 ≈0.323, ROC AUC ≈0.57 (`model_outputs/test_metrics.json`).  
  - Random forest (test): accuracy ≈0.737, churn recall ≈0.132, F1 ≈0.168, ROC AUC ≈0.538 (`model_outputs/rf_test_metrics.json`).  
  - XGBoost (test): accuracy ≈0.644, churn recall ≈0.274, F1 ≈0.236, ROC AUC ≈0.519 (`model_outputs/xgb_test_metrics.json`).  
  Logistic delivers the strongest recall/ROC, while the tree models offer higher accuracy if lower recall is acceptable.

## Model Interpretability
- **Logistic regression analysis (`interpretability/logistic_regression_analysis.py`):** loads the trained logistic pipeline, extracts standardized coefficients, and saves both `interpretability_outputs/logistic_regression/coefficients.tsv` (full ranking) and `top_coefficients.png` (top-N view).
- **Key insights:** the strongest churn reducers are the chronic-condition flags (`has_Z713`, `has_I10`, `has_E119`) and higher app engagement metrics, while signals like irregular web cadence (`web_interval_std`), higher claim-to-app ratio, or fresher claim activity push risk upward. Lower-magnitude terms such as `low_engagement_flag` and `outreach_x_low_engagement` still appear in the coefficient table, indicating modest penalties for under-engaged members and a small benefit when outreach targets that cohort.

## Outreach Prioritization
- **Ranking script (`src/recommend_outreach.py`):** scores every member with the trained pipeline, sorts by churn probability, and writes `recommendations/ranked_scores.csv`, `top_candidates.csv` (top *n*), `summary.json`, and `lift_plot.png`. By default it selects the top 500 members, but you can pass `--top-n` (absolute count) or `--top-percent` plus custom evaluation cutoffs.
- **Usage:**  
  ```bash
  .venv/bin/python src/recommend_outreach.py \
    --model-path model_outputs/logistic_regression.joblib \
    --features-path feature_outputs/features.csv \
    --output-dir recommendations \
    --top-n 500 \
    --evaluation-counts 100,200,500,1000,2000,3000,5000 \
    --plot-counts 100,200,500,1000,2000,3000,4000,5000
  ```  
  The script reports aggregate stats (avg/median predicted churn, low-engagement share, actual churn rate in train) for multiple outreach sizes and saves a cumulative-gain style plot (`lift_plot.png`) to visualize diminishing returns as `n` grows.
- **Decision guidance:** choose `n` where marginal gains flatten relative to outreach cost/capacity; compare cohorts like top 200 vs 500 vs 1000 and ensure prioritized members align with business priorities (e.g., low-engagement or specific ICD flags).
