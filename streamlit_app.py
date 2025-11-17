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
        if hasattr(model, 'estimators_') and model.estimators_:
            all_preds = [t.predict(X_new)[0] for t in model.estimators_]
            std = float(np.std(all_preds))
    except Exception:
        std = None

    pred = max(100.0, pred)
    summary = {
        "Planting Date — ಬಿತ್ತನೆ ದಿನಾಂಕ": planting_date.strftime('%Y-%m-%d'),
        "Harvest Date — ಕಟುವೆ ದಿನಾಂಕ": harvest_date.strftime('%Y-%m-%d'),
        "Scenario — ಹವಾಮಾನ ಸಂದರ್ಭ": scenario_type,
        "Fertilizer (kg/ha) — ರಾಸಾಯನಿಕ ಪೋಷಕಾಂಶ (ಕೆಜಿ/ಹೆಕ್ಟೇರ್)": input_data['fertilizer_kg_ha'],
        "Irrigation (m³/ha) — ಸಿಂಚನ (ಮೀಟರ್³/ಹೆಕ್ಟೇರ್)": input_data['irrigation_m3_ha'],
        "Generated Rainfall (mm) — ಉತ್ಪತ್ತಿಯಾದ ಮಳೆ (ಮಿಮೀ)": f"{input_data['total_precip_mm']:.0f}",
        "Avg Max Temp (°C) — ಸರಾಸರಿ ಗರಿಷ್ಠ ತಾಪಮಾನ (°ಸೆಲ್ಸಿಯಸ್)": f"{input_data['avg_temp_max_C']:.1f}",
        "Total Sunshine (hrs) — ಒಟ್ಟು ಸೂರ್ಯ ಪ್ರಕಾಶ (ಗಂಟೆಗಳು)": f"{input_data['total_sunshine_h']:.0f}",
    }
    return pred, summary, std


def main():
    st.set_page_config(page_title="Generative AI Farm Digital Twin — ಜನರೇಟಿವ್ AI ಫಾರ್ಮ್ ಡಿಜಿಟಲ್ ಟ್ವಿನ್", layout="wide")
    st.title("🌾 Generative AI-Powered Farm Digital Twin — ಜನರೇಟಿವ್ AI ಚಾಲಿತ ಕೃಷಿ ಡಿಜಿಟಲ್ ಟ್ವಿನ್")

    models = load_models()
    if not models:
        st.error("Models not loaded — ಮಾದರಿಗಳು ಲೋಡ್ ಆಗಿಲ್ಲ.")
        return

    st.sidebar.header("🚜 Management Inputs & Scenario Setup — ನಿರ್ವಹಣಾ ಇನ್ಪುಟ್‌ಗಳು ಮತ್ತು ಹವಾಮಾನ ಸ್ಥಿತಿಗತಿ")
    
    selected_crop = st.sidebar.selectbox("1. Select Crop — ಬೆಳೆ ಆಯ್ಕೆ ಮಾಡಿ:", CROPS)
    planting_date = st.sidebar.date_input("2. Select Planting Date — ಬಿತ್ತನೆ ದಿನಾಂಕ ಆಯ್ಕೆ ಮಾಡಿ:", value=datetime(2025,4,1), min_value=datetime(2025,1,1), max_value=datetime(2025,12,31))

    if selected_crop in ['Jowar','Maize']:
        default_fert = 150
        default_irr = 500
    else:
        default_fert = 180
        default_irr = 700

    fertilizer_input = st.sidebar.slider("3. Fertilizer Input (kg/ha) — ರಾಸಾಯನಿಕ ಪೋಷಕಾಂಶ (ಕೆಜಿ/ಹೆಕ್ಟೇರ್):", min_value=50, max_value=300, value=default_fert, step=5)
    irrigation_input = st.sidebar.slider("4. Irrigation Input (m³/ha) — ಸಿಂಚನ (ಮೀಟರ್³/ಹೆಕ್ಟೇರ್):", min_value=100, max_value=2000, value=default_irr, step=50)
    scenario_selection = st.sidebar.selectbox("5. Select Future Weather Scenario — ಭವಿಷ್ಯದ ಹವಾಮಾನ ಸಂದರ್ಭ ಆಯ್ಕೆ ಮಾಡಿ:", SCENARIOS)
    seed_input = st.sidebar.number_input("Random Seed (0 = random) — ಯಾದೃಚ್ಛಿಕ ಬೀಜ (0 = ಯಾದೃಚ್ಛಿಕ)", value=0, min_value=0, step=1)
    seed_val = int(seed_input) if seed_input>0 else None

    management_inputs = {'fertilizer_kg_ha': fertilizer_input, 'irrigation_m3_ha': irrigation_input}

    st.header(f"Results for: {selected_crop} — ಫಲಿತಾಂಶಗಳು")

    if st.sidebar.button("Run Digital Twin Simulation — ಡಿಜಿಟಲ್ ಟ್ವಿನ್ ಸಿಮ್ಯುಲೇಷನ್ ಚಾಲನೆ ಮಾಡಿ"):
        predicted_yield, summary_data, uncertainty = predict_yield_with_uncertainty(
            selected_crop,
            datetime.combine(planting_date, datetime.min.time()),
            management_inputs,
            scenario_selection,
            models,
            seed=seed_val
        )
        if predicted_yield is None:
            st.warning(f"Prediction model for {selected_crop} not available or prediction failed — {selected_crop} ಗಾಗಿ ಮುನ್ಸೂಚನೆ ಮಾದರಿ ಲಭ್ಯವಿಲ್ಲ ಅಥವಾ ಮುನ್ಸೂಚನೆ ವಿಫಲವಾಗಿದೆ.")
            return

        st.markdown(f"## Predicted Yield — ಮುನ್ಸೂಚನೆ ಮಾಡಿದ ಉತ್ಪಾದನೆ:")
        st.success(f"**{predicted_yield:.2f} kg/ha — ಕೆಜಿ/ಹೆಕ್ಟೇರ್**", icon="📈")
        if uncertainty is not None:
            st.write(f"Uncertainty (std across trees) — ಅನುಮಾನ (ವೃಕ್ಷಗಳಲ್ಲಿನ ಸಣ್ಣ ಪ್ರಮಾಣ): {uncertainty:.2f} kg/ha")

        st.write("Generated Rainfall (mm) — ಉತ್ಪತ್ತಿಯಾದ ಮಳೆ (ಮಿಮೀ)")
        st.write("Avg Max Temp (°C) — ಸರಾಸರಿ ಗರಿಷ್ಠ ತಾಪಮಾನ (°ಸೆಲ್ಸಿಯಸ್)")
        st.write("Total Sunshine (hrs) — ಒಟ್ಟು ಸೂರ್ಯ ಪ್ರಕಾಶ (ಗಂಟೆಗಳು)")

        # ------------------ Model Validation & Explainability ------------------
        st.markdown("### Model Validation & Explainability — ಮಾದರಿ ಪರಿಶೀಲನೆ ಮತ್ತು ವಿವರಣೆ")

        metrics_file = f"models/validation_reports/{selected_crop.lower()}_metrics.json"
        shap_img = f"models/validation_reports/{selected_crop.lower()}_shap_summary.png"
        val_csv = f"models/validation_reports/{selected_crop.lower()}_validation.csv"

        alt_meta = os.path.join(MODELS_PATH, f"{selected_crop.lower()}_model.joblib.meta.json")

        metrics = None
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
            except Exception as e:
                st.warning(f"Could not read metrics file — ಮೆಟ್ರಿಕ್ಸ್ ಫೈಲ್ ಓದಲು ಸಾಧ್ಯವಿಲ್ಲ: {e}")
        elif os.path.exists(alt_meta):
            try:
                with open(alt_meta) as f:
                    meta = json.load(f)
                    if 'validation' in meta:
                        metrics = meta['validation']
            except Exception as e:
                st.warning(f"Could not read model metadata — ಮಾದರಿ ಮೆಟಾಡೇಟಾ ಓದಲು ಸಾಧ್ಯವಿಲ್ಲ: {e}")

        if metrics is not None:
            st.write("**Validation metrics (test set) — ಪರಿಶೀಲನೆ ಮೆಟ್ರಿಕ್ಸ್ (ಪರೀಕ್ಷಾ ಸೆಟ್)**")
            st.json(metrics)
        else:
            st.info("Validation metrics not found — ಪರಿಶೀಲನೆ ಮೆಟ್ರಿಕ್ಸ್ ಲಭ್ಯವಿಲ್ಲ. ದಯವಿಟ್ಟು ತರಬೇತಿಯನ್ನು ನಡೆಸಿ.")

        if os.path.exists(shap_img):
            st.image(shap_img, caption="SHAP summary (global feature importance) — SHAP ಸಾರಾಂಶ (ವಿಶ್ವ ವೈಶಿಷ್ಟ್ಯ ಮಹತ್ವ)")
        else:
            st.info("SHAP summary not available — SHAP ಸಾರಾಂಶ ಲಭ್ಯವಿಲ್ಲ.")

        if os.path.exists(val_csv):
            try:
                df_val = pd.read_csv(val_csv)
                st.write("Sample of validation predictions (first 10 rows) — ಪರಿಶೀಲನೆ ಮುನ್ಸೂಚನೆ ಮಾದರಿಗಳ ಮಾದರಿ (ಮೊದಲ 10 ಸಾಲುಗಳು)")
                st.dataframe(df_val.head(10))
            except Exception as e:
                st.warning(f"Could not read validation CSV — ಪರಿಶೀಲನೆ CSV ಓದಲು ಸಾಧ್ಯವಿಲ್ಲ: {e}")
        else:
            st.info("Validation CSV not found — ಪರಿಶೀಲನೆ CSV ಲಭ್ಯವಿಲ್ಲ.")
        # ------------------ End Model Validation & Explainability ------------------

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Scenario Inputs Used — ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ಉಪಯೋಗಿಸಿದ ಇನ್ಪುಟ್‌ಗಳು")
            st.table(pd.Series(summary_data).to_frame('Value — ಮೌಲ್ಯ'))
        with col2:
            st.subheader("Management 'What-If' Comparison (Against Normal) — ನಿರ್ವಹಣೆ 'ಎಲ್ಲಿ' ಹೋಲಿಕೆ (ಸಾಮಾನ್ಯದ ವಿರುದ್ಧ)")
            yield_base, _, _ = predict_yield_with_uncertainty(selected_crop, datetime.combine(planting_date, datetime.min.time()), management_inputs, 'NORMAL', models, seed=seed_val)
            optimized_inputs = {'fertilizer_kg_ha': fertilizer_input * 1.2, 'irrigation_m3_ha': irrigation_input * 1.5}
            yield_optimized, _, _ = predict_yield_with_uncertainty(selected_crop, datetime.combine(planting_date, datetime.min.time()), optimized_inputs, 'NORMAL', models, seed=seed_val)
            comparison_data = {
                "Scenario — ಪರಿಸ್ಥಿತಿ": ["Base (Normal Weather) — ಮೂಲ (ಸಾಮಾನ್ಯ ಹವಾಮಾನ)", "Optimized (Normal Weather) — ಶ್ರೇಷ್ಠೀಕೃತ (ಸಾಮಾನ್ಯ ಹವಾಮಾನ)"],
                "Predicted Yield (kg/ha) — ಮುನ್ಸೂಚನೆ ಮಾಡಿದ ಉತ್ಪಾದನೆ (ಕೆಜಿ/ಹೆಕ್ಟೇರ್)": [f"{yield_base:.2f}", f"{yield_optimized:.2f}"]
            }
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, hide_index=True)
            st.markdown(f"**Difference (Optimized vs Base) — ವ್ಯತ್ಯಾಸ (ಶ್ರೇಷ್ಠೀಕೃತ vs ಮೂಲ):** **{yield_optimized - yield_base:.2f} kg/ha**")

    # --------- Step 10: Optimize for Farmers ---------
    st.markdown("---")
    st.header("🔧 Step 10: Optimize for Farmers — ಕೃಷಿಕರಿಗಾಗಿ ಪರಿಪೂರ್ಣತೆ")

    if st.button("Find Optimal Fertilizer & Irrigation Inputs — ಪರಿಪೂರ್ಣ ರಾಸಾಯನಿಕ ಮತ್ತು ಸಿಂಚನ ಇನ್ಪುಟ್‌ಗಳನ್ನು ಹುಡುಕಿ"):
        best_yield = -np.inf
        best_fert = None
        best_irr = None

        fert_range = range(50, 301, 10)       # 50 to 300 kg/ha by 10
        irr_range = range(100, 2001, 100)    # 100 to 2000 m3/ha by 100

        progress_bar = st.progress(0)
        total_steps = len(fert_range) * len(irr_range)
        step_count = 0

        for fert in fert_range:
            for irr in irr_range:
                test_inputs = {'fertilizer_kg_ha': fert, 'irrigation_m3_ha': irr}
                pred_yield, _, _ = predict_yield_with_uncertainty(
                    selected_crop,
                    datetime.combine(planting_date, datetime.min.time()),
                    test_inputs,
                    scenario_selection,
                    models,
                    seed=seed_val
                )
                if pred_yield is not None and pred_yield > best_yield:
                    best_yield = pred_yield
                    best_fert = fert
                    best_irr = irr

                step_count += 1
                progress_bar.progress(step_count / total_steps)

        st.success(f"Optimal Fertilizer: {best_fert} kg/ha — ಪರಿಪೂರ್ಣ ರಾಸಾಯನಿಕ ಪೋಷಕಾಂಶ")
        st.success(f"Optimal Irrigation: {best_irr} m³/ha — ಪರಿಪೂರ್ಣ ಸಿಂಚನ ಪ್ರಮಾಣ")
        st.success(f"Predicted Yield with Optimized Inputs: {best_yield:.2f} kg/ha — ಪರಿಪೂರ್ಣ ಇನ್ಪುಟ್‌ಗಳೊಂದಿಗೆ ಮುನ್ಸೂಚನೆ ಮಾಡಿದ ಉತ್ಪಾದನೆ")

        current_yield, _, _ = predict_yield_with_uncertainty(
            selected_crop,
            datetime.combine(planting_date, datetime.min.time()),
            management_inputs,
            scenario_selection,
            models,
            seed=seed_val
        )
        if current_yield is not None:
            diff = best_yield - current_yield
            st.info(f"Improvement over current inputs — ಪ್ರಸ್ತುತ ಇನ್ಪುಟ್‌ಗಳಿಗಿಂತ ಸುಧಾರಣೆ: {diff:.2f} kg/ha")


if __name__ == "__main__":
    main()
