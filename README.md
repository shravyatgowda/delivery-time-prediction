# Delivery Time Prediction Engine

A last-mile delivery time prediction pipeline built for a quick-commerce
context: given a pickup/drop location, order time, weather, and traffic
conditions, predict how long the delivery will take.

## Approach

1. **`generate_data.py`** — generates a synthetic but realistic dataset
   (52,000 orders) with an underlying ground-truth time model driven by
   distance, time-of-day peak effects, weather, traffic density, rider
   experience, and order size, plus noise.
2. **`train_model.py`** —
   - Engineers features: haversine distance, cyclical hour/day-of-week
     encoding, one-hot weather/traffic, peak-hour and item-count signals.
   - Trains a **naive baseline** (linear regression on distance alone) to
     establish a reference error.
   - Trains a **tuned XGBoost regressor** over the full feature set, using
     grid search (3-fold CV) over depth, learning rate, subsample ratio,
     and number of estimators.
   - Reports RMSE / MAE for both and prints feature importances.

## Results (this repo's synthetic dataset)

| Model | RMSE (min) | MAE (min) |
|---|---|---|
| Baseline (distance-only linear regression) | 3.59 | 2.88 |
| Tuned XGBoost (full feature set) | 1.83 | 1.46 |

**~49% RMSE reduction** from feature engineering + tuning over the naive
baseline. Traffic density, peak-hour flag, and weather conditions were the
next most important predictors after raw distance.

> Note: this repository uses a synthetic dataset for reproducibility.
> Results on a real operational dataset will differ in absolute minutes
> but the relative gain from feature engineering + tuning follows the
> same pattern.

`data/sample_delivery_data.csv` has a 300-row sample for quick inspection.
Run `generate_data.py` to produce the full 52,000-row dataset used for
training (excluded from the repo — regenerated locally, not committed).

## Run it yourself

```bash
pip install -r requirements.txt
python generate_data.py
python train_model.py
```

## Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost
