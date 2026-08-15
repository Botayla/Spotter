# Spotter — Load Rate Prediction

A machine learning solution for predicting freight `posted_rate` from load
characteristics (pickup/delivery location, distance, weight, equipment type,
market signals, and date), built for the Spotter Machine Learning Engineer
assessment.

The notebook covers the full pipeline: data exploration, data-quality
handling, feature engineering, model comparison, hyperparameter tuning, and
final predictions on the held-out validation set.

## Table of contents

- [Project structure](#project-structure)
- [Setup](#setup)
- [Data](#data)
- [How to run](#how-to-run)
- [Approach](#approach)
- [Results](#results)

## Project structure

```
.
├── README.md
├── requirements.txt
├── notebook/
│   └── Spotter_code.ipynb        # full solution: EDA → cleaning → features → modeling → predictions
├── score.py                      # provided by Spotter: validates submission files and renders the December chart
├── data/                         # not committed — place the provided CSVs here (see Data below)
│   ├── train_test.csv
│   ├── validation.csv
│   ├── validation_predictions_template.csv
│   └── december_chart_inputs.csv
└── outputs/
    ├── validation_predictions.csv       # final required deliverable (load_id, predicted_rate)
    └── december_chart_inputs.csv        # filled in, used by score.py for the December chart
```

## Setup

```bash
git clone <this-repo-url>
cd <this-repo>
pip install -r requirements.txt
```

## Data

The CSV files provided by Spotter are not committed to this repository.
Place them locally in a `data/` folder at the repo root before running the
notebook:

- `data/train_test.csv` — labeled development data
- `data/validation.csv` — 12,000 loads requiring final predictions
- `data/validation_predictions_template.csv` — template to fill with predictions
- `data/december_chart_inputs.csv` — fixed-scenario inputs for the December chart

## How to run

1. Open `notebook/Spotter_code.ipynb` in Jupyter, VS Code, or Google Colab (preferred because I used it and the paths will work without any problems) .
2. Run all cells from top to bottom. The notebook will:
   - explore and clean `train_test.csv` (missing values, invalid entries, duplicates)
   - engineer features: date parts, pickup→delivery bearing angle, one-hot
     encoded equipment type
   - apply a **time-based** train/validation split (holding out the most
     recent two months) rather than a random split, to honestly simulate
     forecasting a genuinely future month like December
   - compare several regression models and tune the strongest candidate
     with `RandomizedSearchCV`
   - persist the trained model, scaler, and expected column order
     (`gbr_final_model.pkl`, `robust_scaler.pkl`, `train_columns.pkl`)
   - predict on `validation.csv`, fill `validation_predictions_template.csv`,
     and save the result as `validation_predictions.csv`
   - reconstruct the missing features for the fixed December scenario in
     `december_chart_inputs.csv`, predict, and run `score.py` to validate
     both output files and render `scorer_results/candidate_december.png`
3. Copy the final `validation_predictions.csv` and the filled
   `december_chart_inputs.csv` into `outputs/` for submission.

## Approach

**Data quality**
- `weight` had a small number of missing and negative values; negatives were
  corrected and missing values imputed with the overall median (no
  relationship found with equipment type or date).
- `market_index` had missing values imputed using the median for that
  specific date, since it behaves as a day-level market signal rather than a
  per-load one.

**Feature engineering**
- Pickup/delivery coordinates, distance, weight, `market_index`, `quote_signal`
- Date parts: month, day-of-week, day-of-month
- A bearing angle capturing trip direction between pickup and delivery
- One-hot encoded equipment type

**Validation strategy**
A time-based split — training on earlier months and validating on the most
recent two — was used instead of a random split. Since the real target is
forecasting December (a month the model has never seen), a random split
would overstate how well the model generalizes to genuinely future data.

**Modeling**
Linear Regression, SVR, Decision Tree, Random Forest, XGBoost, LightGBM,
AdaBoost, Gradient Boosting, Extra Trees, and Stacking/Voting ensembles were
compared on R², MAE, and RMSE. Gradient Boosting was selected as the final
model and tuned with `RandomizedSearchCV` over depth, learning rate, number
of estimators, and subsampling.

**December fixed-scenario chart**
`december_chart_inputs.csv` only provides city names, fixed load attributes,
and dates — not the coordinates or market signals the model was trained on.
These were reconstructed rather than guessed:
- pickup/delivery coordinates were looked up from `train_test.csv`, since
  both cities already appear there with known coordinates
- `market_index` and `quote_signal` were taken from the real per-date median
  values already present in `validation.csv` for November/December, rather
  than extrapolated

## Results

A full write-up of the exploratory findings, data-quality issues, validation
methodology, and model performance is provided separately in the
accompanying report (PDF/DOCX), including the fixed December prediction
chart produced by `score.py`.
