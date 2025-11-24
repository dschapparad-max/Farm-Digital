#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os

RAW_DATA_PATH = 'data/raw'
PROCESSED_DATA_PATH = 'data/processed'
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

weather_file = os.path.join(RAW_DATA_PATH, 'weather_historical.csv')
farm_file = os.path.join(RAW_DATA_PATH, 'farm_records_simulated.csv')

# --- Load and clean weather data ---
df_weather = pd.read_csv(weather_file, parse_dates=['date'])
df_weather = df_weather.sort_values('date').reset_index(drop=True)

# Ensure daily frequency and fill missing days
df_weather = df_weather.set_index('date').asfreq('D')

# Interpolate interior gaps by time, then fill any remaining edge NaNs
df_weather = df_weather.interpolate(method='time').bfill().ffill()

df_weather = df_weather.reset_index()

# Basic validation: required columns
required_weather_cols = {'precip_mm', 'sunshine_h', 'temp_max_C', 'temp_min_C'}
missing_cols = required_weather_cols - set(df_weather.columns)
if missing_cols:
    raise RuntimeError(f"Missing required weather columns: {missing_cols}")

# Cumulative sums for fast interval aggregation
df_weather['precip_cum'] = df_weather['precip_mm'].cumsum()
df_weather['sun_cum'] = df_weather['sunshine_h'].cumsum()
df_weather['tempmax_cum'] = df_weather['temp_max_C'].cumsum()
df_weather['tempmin_cum'] = df_weather['temp_min_C'].cumsum()

dates = df_weather['date'].values              # numpy datetime64 array
precips = df_weather['precip_cum'].values
suns = df_weather['sun_cum'].values
tmaxs = df_weather['tempmax_cum'].values
tmins = df_weather['tempmin_cum'].values
n_days = len(dates)

# --- Load farm simulated data ---
df_farm = pd.read_csv(farm_file, parse_dates=['planting_date','harvest_date'])

def agg_features_row(planting, harvest):
    """
    Return: (precip_sum, avg_tmax, avg_tmin, sun_sum)
    Returns NaNs if dates are invalid or out-of-range.
    """
    if pd.isna(planting) or pd.isna(harvest) or (harvest < planting):
        return (np.nan, np.nan, np.nan, np.nan)
    p64 = np.datetime64(planting)
    h64 = np.datetime64(harvest)
    p_idx = np.searchsorted(dates, p64, side='left')
    h_idx = np.searchsorted(dates, h64, side='right') - 1
    # Validate indices
    if p_idx < 0 or h_idx < 0 or p_idx >= n_days or h_idx >= n_days or h_idx < p_idx:
        return (np.nan, np.nan, np.nan, np.nan)
    precip_sum = precips[h_idx] - (precips[p_idx-1] if p_idx > 0 else 0.0)
    sun_sum = suns[h_idx] - (suns[p_idx-1] if p_idx > 0 else 0.0)
    tmax_sum = tmaxs[h_idx] - (tmaxs[p_idx-1] if p_idx > 0 else 0.0)
    tmin_sum = tmins[h_idx] - (tmins[p_idx-1] if p_idx > 0 else 0.0)
    days = (h_idx - p_idx + 1)
    avg_tmax = tmax_sum / days
    avg_tmin = tmin_sum / days
    return (precip_sum, avg_tmax, avg_tmin, sun_sum)

# Compute aggregated features (still fast: cumsum lookups are O(1) per row)
agg_results = df_farm.apply(lambda r: agg_features_row(r['planting_date'], r['harvest_date']), axis=1)
agg_df = pd.DataFrame(agg_results.tolist(), columns=[
    'total_precip_mm', 'avg_temp_max_C', 'avg_temp_min_C', 'total_sunshine_h'
])

df_final = pd.concat([df_farm.reset_index(drop=True), agg_df], axis=1)

# --- Save per-crop processed CSVs for modeling ---
FEATURES = ['fertilizer_kg_ha', 'pesticide_l_ha', 'total_precip_mm', 'avg_temp_max_C', 'total_sunshine_h']
TARGET = 'yield_kg_ha'

for crop in df_final['crop'].unique():
    df_crop = df_final[df_final['crop'] == crop].copy()
    df_crop_model = df_crop[['planting_date','harvest_date'] + FEATURES + [TARGET]].dropna()
    out_path = os.path.join(PROCESSED_DATA_PATH, f"{crop.lower()}_model_data.csv")
    df_crop_model.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df_crop_model)} rows)")
