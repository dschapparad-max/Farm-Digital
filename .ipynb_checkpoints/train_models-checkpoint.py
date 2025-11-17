#!/usr/bin/env python
# coding: utf-8

# In[2]:


# File: models/train_models.py
# Updated: Adds detailed validation reporting (MAE, RMSE, R2, 95% PI, coverage)

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# ---------------------------
# Paths and config
# ---------------------------
RAW_PROCESSED = 'data/processed'
MODELS_DIR = 'models'
VALIDATION_DIR = os.path.join(MODELS_DIR, 'validation_reports')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VALIDATION_DIR, exist_ok=True)

CROPS = ['Jowar', 'Paddy', 'Maize', 'Cotton']

FEATURES = [
    'fertilizer_kg_ha',
    'irrigation_m3_ha',
    'total_precip_mm',
    'avg_temp_max_C',
    'total_sunshine_h'
]

TARGET = 'yield_kg_ha'
RANDOM_STATE = 42

# Helper: save JSON
def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)

# ---------------------------
# Training + Validation Loop
# ---------------------------
for crop in CROPS:
    print(f"\n=== Training & Validation for: {crop} ===")

    csv_path = os.path.join(RAW_PROCESSED, f"{crop.lower()}_model_data.csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {crop}: {csv_path} not found")
        continue

    df = pd.read_csv(csv_path)
    if len(df) < 10:
        print(f"Skipping {crop}: not enough rows ({len(df)})")
        continue

    X = df[FEATURES].values
    y = df[TARGET].values

    # Split: use a hold-out test set for validation reporting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Basic metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Estimate uncertainty from trees (std across estimators)
    y_pred_tree_matrix = None
    pred_std = None
    if hasattr(model, 'estimators_') and len(model.estimators_) > 1:
        try:
            # each tree's predictions on X_test -> shape (n_trees, n_samples)
            y_pred_tree_matrix = np.vstack([t.predict(X_test) for t in model.estimators_])
            pred_std = np.std(y_pred_tree_matrix, axis=0, ddof=0)  # per-sample std
            # 95% prediction interval via mean +/- 1.96*std
            lower_95 = y_pred - 1.96 * pred_std
            upper_95 = y_pred + 1.96 * pred_std
            # coverage: fraction of true y inside the interval
            coverage = float(np.mean((y_test >= lower_95) & (y_test <= upper_95)))
        except Exception as e:
            print("Warning: could not compute tree-based std:", e)
            pred_std = np.full_like(y_pred, np.nan)
            lower_95 = np.full_like(y_pred, np.nan)
            upper_95 = np.full_like(y_pred, np.nan)
            coverage = None
    else:
        pred_std = np.full_like(y_pred, np.nan)
        lower_95 = np.full_like(y_pred, np.nan)
        upper_95 = np.full_like(y_pred, np.nan)
        coverage = None

    # Build validation dataframe and save
    val_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred_mean': y_pred,
        'y_pred_std': pred_std,
        'lower_95': lower_95,
        'upper_95': upper_95,
        'residual': y_test - y_pred
    })

    val_csv_path = os.path.join(VALIDATION_DIR, f"{crop.lower()}_validation.csv")
    val_df.to_csv(val_csv_path, index=False)
    print(f"Saved validation CSV: {val_csv_path} ({len(val_df)} rows)")

    # Save simple scatter plot (y_true vs y_pred)
    try:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(y_test, y_pred, alpha=0.6, s=10)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=1)  # identity line
        ax.set_xlabel('y_true (kg/ha)')
        ax.set_ylabel('y_pred (kg/ha)')
        ax.set_title(f"{crop} - True vs Pred (MAE={mae:.2f})")
        plt.tight_layout()
        plot_path = os.path.join(VALIDATION_DIR, f"{crop.lower()}_true_vs_pred.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved scatter plot: {plot_path}")
    except Exception as e:
        print("Warning: could not save plot:", e)

    # Save metrics JSON
    metrics = {
        'crop': crop,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'prediction_interval_95_coverage': coverage if coverage is not None else None
    }
    metrics_path = os.path.join(VALIDATION_DIR, f"{crop.lower()}_metrics.json")
    save_json(metrics, metrics_path)
    print(f"Saved metrics JSON: {metrics_path}")

    # Save model + metadata (embed metrics into metadata)
    model_path = os.path.join(MODELS_DIR, f"{crop.lower()}_model.joblib")
    joblib.dump(model, model_path)

    metadata = {
        "features": FEATURES,
        "units": {
            "fertilizer_kg_ha": "kg/ha",
            "irrigation_m3_ha": "m3/ha",
            "total_precip_mm": "mm",
            "avg_temp_max_C": "degC",
            "total_sunshine_h": "hours"
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "validation_metrics": metrics
    }
    meta_file = model_path + '.meta.json'
    save_json(metadata, meta_file)
    print(f"Saved model and metadata to: {model_path} and {meta_file}")

print("\nAll training + validation reports complete.")


# In[3]:


