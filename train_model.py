"""
Trains and evaluates a delivery-time prediction model.

Compares a naive distance-only baseline against a feature-engineered,
tuned XGBoost regressor, and reports RMSE / MAE for both.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

df = pd.read_csv("data/delivery_data.csv")

# ---------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------
df["hour_sin"] = np.sin(2 * np.pi * df["order_hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["order_hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

df = pd.get_dummies(df, columns=["weather", "traffic_density"], drop_first=True)

feature_cols = [
    "distance_km", "is_weekend", "is_peak_hour", "rider_experience_days",
    "num_items", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
] + [c for c in df.columns if c.startswith("weather_") or c.startswith("traffic_density_")]

X = df[feature_cols]
y = df["delivery_time_min"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------------
# Baseline: distance-only linear regression
# ---------------------------------------------------------------
baseline = LinearRegression()
baseline.fit(X_train[["distance_km"]], y_train)
baseline_preds = baseline.predict(X_test[["distance_km"]])
baseline_rmse = mean_squared_error(y_test, baseline_preds) ** 0.5
baseline_mae = mean_absolute_error(y_test, baseline_preds)

# ---------------------------------------------------------------
# Tuned XGBoost regressor on the full engineered feature set
# ---------------------------------------------------------------
param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
search = GridSearchCV(xgb_model, param_grid, cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
search.fit(X_train, y_train)

best_model = search.best_estimator_
tuned_preds = best_model.predict(X_test)
tuned_rmse = mean_squared_error(y_test, tuned_preds) ** 0.5
tuned_mae = mean_absolute_error(y_test, tuned_preds)

# ---------------------------------------------------------------
# Results
# ---------------------------------------------------------------
print("=" * 55)
print("BASELINE (distance-only Linear Regression)")
print(f"  RMSE: {baseline_rmse:.2f} min   MAE: {baseline_mae:.2f} min")
print()
print("TUNED XGBoost (full engineered feature set)")
print(f"  Best params: {search.best_params_}")
print(f"  RMSE: {tuned_rmse:.2f} min   MAE: {tuned_mae:.2f} min")
print()
improvement = (baseline_rmse - tuned_rmse) / baseline_rmse * 100
print(f"RMSE improvement over baseline: {improvement:.1f}%")
print("=" * 55)

# feature importance
importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nTop feature importances:")
print(importances.head(8))
