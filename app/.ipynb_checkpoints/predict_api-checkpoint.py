# File: app/predict_api.py
from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import json
from datetime import datetime, timedelta
import pandas as pd

app = Flask(_name_)

MODELS_DIR = os.environ.get('MODELS_DIR', 'models')  # configurable
SUPPORTED_SCENARIOS = {'NORMAL','DROUGHT','HEAVY_RAIN'}

MODELS = {}
METADATA = {}
if os.path.exists(MODELS_DIR):
    for fname in os.listdir(MODELS_DIR):
        if fname.endswith('.joblib'):
            crop = fname.replace('_model.joblib','').capitalize()
            try:
                model = joblib.load(os.path.join(MODELS_DIR, fname))
                MODELS[crop] = model
                meta_path = os.path.join(MODELS_DIR, f'{crop.lower()}_model.joblib.meta.json')
                if os.path.exists(meta_path):
                    try:
                        METADATA[crop] = json.load(open(meta_path))
                    except Exception:
                        METADATA[crop] = None
                else:
                    METADATA[crop] = None
                app.logger.info(f"Loaded model for {crop}")
            except Exception as e:
                app.logger.error(f"Failed to load model {fname}: {e}")


def _validate_request_json(data):
    required = ['crop','planting_date','fertilizer_kg_ha','irrigation_m3_ha','scenario_type']
    missing = [k for k in required if k not in data]
    if missing:
        return False, f"Missing required fields: {missing}"
    if data['scenario_type'] not in SUPPORTED_SCENARIOS:
        return False, f"Unsupported scenario_type: {data['scenario_type']}"
    try:
        datetime.strptime(data['planting_date'], '%Y-%m-%d')
    except Exception:
        return False, "planting_date must be YYYY-MM-DD"
    try:
        float(data['fertilizer_kg_ha'])
        float(data['irrigation_m3_ha'])
    except Exception:
        return False, "fertilizer_kg_ha and irrigation_m3_ha must be numeric"
    return True, None


def generate_future_weather_scenario(duration_days, scenario_type='NORMAL', seed=None):
    rng = np.random.default_rng(seed)
    base_precip = 4.5 * duration_days
    base_temp_max = 28.0
    base_sunshine = 7.5 * duration_days

    if scenario_type == 'DROUGHT':
        precip_factor = 0.4
        temp_max_adjustment = 3.5
    elif scenario_type == 'HEAVY_RAIN':
        precip_factor = 1.6
        temp_max_adjustment = -1.0
    else:
        precip_factor = 1.0
        temp_max_adjustment = 0.0

    scenario_data = {
        'total_precip_mm': max(0, base_precip * precip_factor + float(rng.uniform(-50,50))),
        'avg_temp_max_C': float(base_temp_max + temp_max_adjustment + float(rng.uniform(-1.0,1.0))),
        'total_sunshine_h': max(0, base_sunshine * (1 + float(rng.uniform(-0.1,0.1))))
    }
    return scenario_data


@app.route('/predict_yield', methods=['POST'])
def predict_yield():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error":"Invalid or missing JSON"}), 400

    ok, msg = _validate_request_json(payload)
    if not ok:
        return jsonify({"error": msg}), 400

    crop = payload['crop']
    crop_key = crop.capitalize()
    model = MODELS.get(crop_key)
    if model is None:
        return jsonify({"error": f"No model loaded for crop: {crop}"}), 404

    planting_date = datetime.strptime(payload['planting_date'], '%Y-%m-%d')
    fertilizer = float(payload['fertilizer_kg_ha'])
    irrigation = float(payload['irrigation_m3_ha'])
    scenario = payload['scenario_type']
    seed = payload.get('random_seed', None)
    try:
        duration_days = int(payload.get('duration_days', 150))
    except Exception:
        duration_days = 150

    weather = generate_future_weather_scenario(duration_days, scenario, seed)

    meta = METADATA.get(crop_key) or {}
    features_order = meta.get('features') if meta.get('features') else ['fertilizer_kg_ha','irrigation_m3_ha','total_precip_mm','avg_temp_max_C','total_sunshine_h']
    input_vector = {}
    for f in features_order:
        if f == 'fertilizer_kg_ha': input_vector[f] = fertilizer
        elif f == 'irrigation_m3_ha': input_vector[f] = irrigation
        else:
            input_vector[f] = weather.get(f)
    X_new = pd.DataFrame([input_vector], columns=features_order)

    try:
        y_pred = float(model.predict(X_new)[0])
    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({"error":"Model prediction failed"}), 500

    # ensure minimum sensible yield
    predicted_yield = max(100.0, y_pred)

    # Build validation for the SINGLE prediction
    uncertainty_std = None
    lower_95 = None
    upper_95 = None

    try:
        if hasattr(model, 'estimators_') and len(model.estimators_) > 1:
            tree_preds = np.array([t.predict(X_new)[0] for t in model.estimators_])
            # population std (ddof=0) matches how many random forest implementations report spread
            uncertainty_std = float(np.std(tree_preds, ddof=0))
            lower_95 = float(predicted_yield - 1.96 * uncertainty_std)
            upper_95 = float(predicted_yield + 1.96 * uncertainty_std)
    except Exception:
        uncertainty_std = None
        lower_95 = None
        upper_95 = None

    # Load model-level validation metrics (saved during training)
    model_validation = meta.get("validation_metrics", {}) if meta else {}

    # Final API response
    response = {
        "crop": crop_key,
        "predicted_yield_kg_ha": round(predicted_yield, 2),

        # NEW: validation of this specific prediction
        "validation_for_prediction": {
            "uncertainty_std": round(uncertainty_std, 2) if uncertainty_std is not None else None,
            "lower_95": round(lower_95, 2) if lower_95 is not None else None,
            "upper_95": round(upper_95, 2) if upper_95 is not None else None
        },

        # NEW: validation metrics from training
        "model_validation_metrics": {
            "mae": model_validation.get("mae"),
            "rmse": model_validation.get("rmse"),
            "r2": model_validation.get("r2")
        },

        "scenario_details": {
            "scenario_type": scenario,
            "planting_date": payload['planting_date'],
            "harvest_date": (planting_date + timedelta(days=duration_days)).strftime('%Y-%m-%d'),
            "fertilizer_kg_ha": fertilizer,
            "irrigation_m3_ha": irrigation,
            "total_precip_mm": round(weather['total_precip_mm'], 1),
            "avg_temp_max_C": round(weather['avg_temp_max_C'], 2),
            "total_sunshine_h": round(weather['total_sunshine_h'], 1)
        }
    }
    return jsonify(response), 200


if _name_ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=False)