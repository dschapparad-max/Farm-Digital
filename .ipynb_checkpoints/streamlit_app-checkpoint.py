import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta

MODELS_PATH = 'models/'
CROPS = ['Jowar', 'Paddy', 'Maize', 'Cotton']
SCENARIOS = ['NORMAL', 'DROUGHT', 'HEAVY_RAIN']
FEATURES = ['fertilizer_kg_ha', 'irrigation_m3_ha', 'total_precip_mm', 'avg_temp_max_C', 'total_sunshine_h']
GROWING_PERIODS = {'Jowar':128,'Paddy':165,'Maize':114,'Cotton':200}

@st.cache_resource
def load_models():
    models = {}
    for crop in CROPS:
        model_path = os.path.join(MODELS_PATH, f'{crop.lower()}_model.joblib')
        if os.path.exists(model_path):
            try:
                models[crop] = joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading model for {crop}: {e}")
        else:
            st.error(f"Model for {crop} not found at {model_path}. Please run training.")
    return models


def generate_future_weather_scenario(duration_days, scenario_type, seed=None):
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


def predict_yield_with_uncertainty(crop, planting_date, management_inputs, scenario_type, models, seed=None):
    if crop not in models:
        return None, None, None
    model = models[crop]
    duration_days = GROWING_PERIODS.get(crop,150)
    harvest_date = planting_date + timedelta(days=duration_days)
    weather_features = generate_future_weather_scenario(duration_days, scenario_type, seed)
    input_data = {
        'fertilizer_kg_ha': management_inputs['fertilizer_kg_ha'],
        'irrigation_m3_ha': management_inputs['irrigation_m3_ha'],
        **weather_features
    }
    X_new = pd.DataFrame([input_data], columns=FEATURES)
    try:
        pred = float(model.predict(X_new)[0])
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        return None, None, None

    std = None
    try:
        # Best-effort: if sklearn ensemble-like object with estimators_
        if hasattr(model, 'estimators_') and model.estimators_:
            all_preds = [t.predict(X_new)[0] for t in model.estimators_]
            std = float(np.std(all_preds))
    except Exception:
        std = None

    pred = max(100.0, pred)
    summary = {
        "Planting Date": planting_date.strftime('%Y-%m-%d'),
        "Harvest Date": harvest_date.strftime('%Y-%m-%d'),
        "Scenario": scenario_type,
        "Fertilizer (kg/ha)": input_data['fertilizer_kg_ha'],
        "Irrigation (m³/ha)": input_data['irrigation_m3_ha'],
        "Generated Rainfall (mm)": f"{input_data['total_precip_mm']:.0f}",
        "Avg Max Temp (°C)": f"{input_data['avg_temp_max_C']:.1f}",
        "Total Sunshine (hrs)": f"{input_data['total_sunshine_h']:.0f}",
    }
    return pred, summary, std


def main():
    st.set_page_config(page_title="Generative AI Farm Digital Twin", layout="wide")
    st.title("🌾 Generative AI-Powered Farm Digital Twin")
    models = load_models()
    if not models:
        st.error("Models not loaded.")
        return

    st.sidebar.header("🚜 Management Inputs & Scenario Setup")

    selected_crop = st.sidebar.selectbox("1. Select Crop:", CROPS)
    planting_date = st.sidebar.date_input("2. Select Planting Date:", value=datetime(2025,4,1), min_value=datetime(2025,1,1), max_value=datetime(2025,12,31))
    if selected_crop in ['Jowar','Maize']:
        default_fert = 150
        default_irr = 500
    else:
        default_fert = 180
        default_irr = 700

    fertilizer_input = st.sidebar.slider("3. Fertilizer Input (kg/ha):", min_value=50, max_value=300, value=default_fert, step=5)
    irrigation_input = st.sidebar.slider("4. Irrigation Input (m³/ha):", min_value=100, max_value=2000, value=default_irr, step=50)
    scenario_selection = st.sidebar.selectbox("5. Select Future Weather Scenario:", SCENARIOS)
    seed_input = st.sidebar.number_input("Random Seed (0 = random)", value=0, min_value=0, step=1)
    seed_val = int(seed_input) if seed_input>0 else None

    management_inputs = {'fertilizer_kg_ha': fertilizer_input, 'irrigation_m3_ha': irrigation_input}

    st.header(f"Results for: {selected_crop}")

    if st.sidebar.button("Run Digital Twin Simulation"):
        predicted_yield, summary_data, uncertainty = predict_yield_with_uncertainty(
            selected_crop,
            datetime.combine(planting_date, datetime.min.time()),
            management_inputs,
            scenario_selection,
            models,
            seed=seed_val
        )
        if predicted_yield is None:
            st.warning(f"Prediction model for {selected_crop} not available or prediction failed.")
            return

        st.markdown(f"## Predicted Yield:")
        st.success(f"**{predicted_yield:.2f} kg/ha**", icon="📈")
        if uncertainty is not None:
            st.write(f"Uncertainty (std across trees): {uncertainty:.2f} kg/ha")

        # ------------------ INSERTED: Model Validation & Explainability ------------------
        st.markdown("### Model Validation & Explainability")

        # Primary expected files (as requested)
        metrics_file = f"models/validation_reports/{selected_crop.lower()}_metrics.json"
        shap_img = f"models/validation_reports/{selected_crop.lower()}_shap_summary.png"
        val_csv = f"models/validation_reports/{selected_crop.lower()}_validation.csv"

        # Fallback: metadata saved next to model (train_models.py saves model.joblib.meta.json)
        alt_meta = os.path.join(MODELS_PATH, f"{selected_crop.lower()}_model.joblib.meta.json")

        # Load metrics (try metrics_file first, else alt_meta)
        metrics = None
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except Exception as e:
                st.warning(f"Could not read metrics file: {e}")
        elif os.path.exists(alt_meta):
            try:
                with open(alt_meta) as f:
                    meta = json.load(f)
                    if 'validation' in meta:
                        metrics = meta['validation']
            except Exception as e:
                st.warning(f"Could not read model metadata: {e}")

        if metrics is not None:
            st.write("**Validation metrics (test set)**")
            st.json(metrics)
        else:
            st.info("Validation metrics not found. Run training to generate them.")

        # SHAP image
        if os.path.exists(shap_img):
            st.image(shap_img, caption="SHAP summary (global feature importance)")
        else:
            st.info("SHAP summary not available.")

        # Validation CSV preview
        if os.path.exists(val_csv):
            try:
                df_val = pd.read_csv(val_csv)
                st.write("Sample of validation predictions (first 10 rows)")
                st.dataframe(df_val.head(10))
            except Exception as e:
                st.warning(f"Could not read validation CSV: {e}")
        else:
            st.info("Validation CSV not found.")
        # ------------------ END INSERTED BLOCK ------------------

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Scenario Inputs Used")
            st.table(pd.Series(summary_data).to_frame('Value'))
        with col2:
            st.subheader("Management 'What-If' Comparison (Against Normal)")
            yield_base, _, _ = predict_yield_with_uncertainty(selected_crop, datetime.combine(planting_date, datetime.min.time()), management_inputs, 'NORMAL', models, seed=seed_val)
            optimized_inputs = {'fertilizer_kg_ha': fertilizer_input * 1.2, 'irrigation_m3_ha': irrigation_input * 1.5}
            yield_optimized, _, _ = predict_yield_with_uncertainty(selected_crop, datetime.combine(planting_date, datetime.min.time()), optimized_inputs, 'NORMAL', models, seed=seed_val)
            comparison_data = {"Scenario": ["Base (Normal Weather)", "Optimized (Normal Weather)"], "Predicted Yield (kg/ha)": [f"{yield_base:.2f}", f"{yield_optimized:.2f}"]}
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, hide_index=True)
            st.markdown(f"**Difference (Optimized vs Base):** **{yield_optimized - yield_base:.2f} kg/ha**")

if __name__ == "__main__":
    main()
