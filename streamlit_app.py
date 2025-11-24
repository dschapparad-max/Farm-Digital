import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from datetime import datetime, timedelta

# ---------------- Config ----------------
MODELS_PATH = 'models/'
CROPS = ['Jowar', 'Paddy', 'Maize', 'Cotton']
SCENARIOS = ['NORMAL', 'DROUGHT', 'HEAVY_RAIN']   # internal codes
FEATURES = ['fertilizer_kg_ha', 'pesticide_l_ha', 'total_precip_mm', 'avg_temp_max_C', 'total_sunshine_h']
GROWING_PERIODS = {'Jowar':128,'Paddy':165,'Maize':114,'Cotton':200}

# ---------------- Language dictionaries ----------------
LANG_DICT = {
    'en': {
        'title': "🌾 Generative AI-Powered Farm Digital Twin",
        'sidebar_header': "🚜 Management Inputs & Scenario Setup",
        'select_crop': "1. Select Crop:",
        'planting_date': "2. Select Planting Date:",
        'fertilizer_input': "3. Fertilizer Input (kg/ha):",
        'pesticide_input': "4. Pesticide Input (L/ha):",
        'scenario_selection': "5. Select Future Weather Scenario:",
        'random_seed': "Random Seed (0 = random)",
        'run_sim': "Run Digital Twin Simulation",
        'predicted_yield': "Predicted Yield",
        'uncertainty': "Uncertainty (std across trees)",
        'model_val': "Model Validation & Explainability",
        'val_metrics': "Validation metrics (test set)",
        'shap_summary': "SHAP summary (global feature importance)",
        'scenario_inputs': "Scenario Inputs Used",
        'comparison': "Management 'What-If' Comparison (Against Normal)",
        'difference': "Difference (Optimized vs Base)",
        'step10_header': "🔧 Step 10: Optimize for Farmers",
        'find_optimal': "Find Optimal Fertilizer & Pesticide Inputs",
        'optimal_fert': "Optimal Fertilizer",
        'optimal_pest': "Optimal Pesticide",
        'predicted_yield_opt': "Predicted Yield with Optimized Inputs",
        'improvement': "Improvement over current inputs",
        'models_not_loaded': "Models not loaded. Please ensure model files exist in 'models/' directory.",
        'optimizing': "Optimizing over fertilizer & pesticide grid..."
    },
    'kn': {
        'title': "🌾 ಜನರೇಟಿವ್ AI ಚಾಲಿತ ಕೃಷಿ ಡಿಜಿಟಲ್ ಟ್ವಿನ್",
        'sidebar_header': "🚜 ನಿರ್ವಹಣಾ ಇನ್ಪುಟ್‌ಗಳು ಮತ್ತು ಹವಾಮಾನ ಸ್ಥಿತಿಗತಿ",
        'select_crop': "1. ಬೆಳೆ ಆಯ್ಕೆ ಮಾಡಿ:",
        'planting_date': "2. ಬೀಜ ದಿವಸ ಆಯ್ಕೆ ಮಾಡಿ:",
        'fertilizer_input': "3. ರಾಸಾಯನಿಕ ಪೋಷಕಾಂಶ (ಕೆಜಿ/ಹೆಕ್ಟೇರ್):",
        'pesticide_input': "4. ಕೀಟನಾಶಕ ಇನ್ಪುಟ್ (L/ha):",
        'scenario_selection': "5. ಭವಿಷ್ಯದ ಹವಾಮಾನ ಕುರಿತು ಆಯ್ಕೆ ಮಾಡಿ:",
        'random_seed': "ಯಾದೃಚ್ಛಿಕ ಬೀಜ (0 = ಯಾದೃಚ್ಛಿಕ)",
        'run_sim': "ಡಿಜಿಟಲ್ ಟ್ವಿನ್ ಸಿಮ್ಯುಲೇಷನ್ ಚಾಲನೆ ಮಾಡಿ",
        'predicted_yield': "ಮುನ್ನನೋಡಿದ ಉತ್ಪಾದನೆ",
        'uncertainty': "ಅನಿಶ್ಚಿತತೆ (ವೃಕ್ಷಗಳ ನಡುವಿನ ಸ್ಟ್ಯಾಂಡರ್ಡ್ ಡೆವಿಯೇಷನ್)",
        'model_val': "ಮಾದರಿ ಪರಿಶೀಲನೆ ಮತ್ತು ವಿವರಣೆ",
        'val_metrics': "ಪರಿಶೀಲನೆ ಮೆಟ್ರಿಕ್ಸ್ (ಪರಿಕ್ಷಾ ಸೆಟ್)",
        'shap_summary': "SHAP ಸಾರಾಂಶ (ಗ್ಲೋಬಲ್ ವೈಶಿಷ್ಟ್ಯ ಮಹತ್ವ)",
        'scenario_inputs': "ಬಳಸಿದ ಪರಿಸ್ಥಿತಿ ಇನ್ಪುಟ್ಗಳು",
        'comparison': "ನಿರ್ವಹಣಾ 'ಎಲ್ಲಿ' ಹೋಲಿಕೆ (ಸಾಮಾನ್ಯ ವಿರುದ್ಧ)",
        'difference': "ವ್ಯತ್ಯಾಸ (ಶ್ರೇಷ್ಠೀಕೃತ vs ಮೂಲ)",
        'step10_header': "🔧 ಕೃಷಿಕರಿಗಾಗಿ ಪರಿಪೂರ್ಣತೆ",
        'find_optimal': "ಪರಿಪೂರ್ಣ ರಾಸಾಯನಿಕ ಮತ್ತು ಕೀಟನಾಶಕ ಇನ್ಪುಟ್‌ಗಳನ್ನು ಹುಡುಕಿ",
        'optimal_fert': "ಪರಿಪೂರ್ಣ ರಾಸಾಯನಿಕ ಪೋಷಕಾಂಶ",
        'optimal_pest': "ಪರಿಪೂರ್ಣ ಕೀಟನಾಶಕ ಪ್ರಮಾಣ",
        'predicted_yield_opt': "ಪರಿಪೂರ್ಣ ಇನ್ಪುಟ್‌ಗಳೊಂದಿಗೆ ಮುನ್ನನೋಡಿದ ಉತ್ಪಾದನೆ",
        'improvement': "ಪ್ರಸ್ತುತ ಇನ್ಪುಟ್‌ಗಳಿಗಿಂತ ಸುಧಾರಣೆ",
        'models_not_loaded': "ಮಾದರಿ ಲೋಡ್ ಆಗುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು 'models/' ಫೋಲ್ಡರ್ ಪರಿಶೀಲಿಸಿ.",
        'optimizing': "ರಾಸಾಯನಿಕ ಮತ್ತು ಕೀಟನಾಶಕ ಗ್ರಿಡ್ ಬಗ್ಗೆ ಪರಿಪೂರ್ಣತೆ ಟ್ರೈ ಮಾಡಲಾಗುತ್ತಿದೆ..."
    }
}

# ---------------- Localized display names ----------------
DISPLAY_CROPS = {
    'en': {'Jowar': 'Jowar', 'Paddy': 'Paddy', 'Maize': 'Maize', 'Cotton': 'Cotton'},
    'kn': {'Jowar': 'ಜೋಳ', 'Paddy': 'ಅಕ್ಕಿ', 'Maize': 'ಮೆಕ್ಕೆ ಜೋಳ', 'Cotton': 'ಹತ್ತಿ'}
}

DISPLAY_SCENARIOS = {
    'en': {'NORMAL': 'NORMAL', 'DROUGHT': 'DROUGHT', 'HEAVY_RAIN': 'HEAVY_RAIN'},
    'kn': {'NORMAL': 'ಸಾಮಾನ್ಯ', 'DROUGHT': 'ಬೂಲಿಭಟ್ಟೆ/ಬಿರಲ್ಲಿಕೆ', 'HEAVY_RAIN': 'ಭಾರೀ ಮಳೆ'}
}
# Note: user can refine Kannada scenario wording; above are reasonable short labels.

# ---------------- Utility: load models ----------------
@st.cache_resource
def load_models():
    models = {}
    for crop in CROPS:
        model_path = os.path.join(MODELS_PATH, f'{crop.lower()}_model.joblib')
        if os.path.exists(model_path):
            try:
                models[crop] = joblib.load(model_path)
            except Exception as e:
                st.warning(f"Could not load model {model_path}: {e}")
    return models

# ---------------- Weather scenario generator ----------------
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

    return {
        'total_precip_mm': max(0.0, base_precip * precip_factor + float(rng.uniform(-50,50))),
        'avg_temp_max_C': float(base_temp_max + temp_max_adjustment + float(rng.uniform(-1.0,1.0))),
        'total_sunshine_h': max(0.0, base_sunshine * (1 + float(rng.uniform(-0.1,0.1))))
    }

# ---------------- Prediction helper ----------------
def predict_yield_with_uncertainty(crop, planting_date, management_inputs, scenario_code, models, seed=None, lang_code='en'):
    """
    Returns: (predicted_yield (float), summary (dict localized), std (float|null))
    """
    if crop not in models or models[crop] is None:
        return None, None, None

    model = models[crop]
    duration = GROWING_PERIODS.get(crop, 150)
    harvest_date = planting_date + timedelta(days=duration)

    weather = generate_future_weather_scenario(duration, scenario_code, seed)
    X = pd.DataFrame([{
        'fertilizer_kg_ha': float(management_inputs.get('fertilizer_kg_ha', 150)),
        'pesticide_l_ha': float(management_inputs.get('pesticide_l_ha', 10.0)),
        'total_precip_mm': weather['total_precip_mm'],
        'avg_temp_max_C': weather['avg_temp_max_C'],
        'total_sunshine_h': weather['total_sunshine_h']
    }], columns=FEATURES)

    try:
        pred = float(model.predict(X)[0])
    except Exception as e:
        # prediction failed for some reason
        return None, None, None

    pred = max(0.0, pred)

    std = None
    try:
        if hasattr(model, 'estimators_') and len(model.estimators_) > 1:
            tree_preds = np.array([t.predict(X)[0] for t in model.estimators_])
            std = float(np.std(tree_preds, ddof=0))
    except Exception:
        std = None

    # localization for labels
    crop_label = DISPLAY_CROPS.get(lang_code, DISPLAY_CROPS['en']).get(crop, crop)
    scenario_label = DISPLAY_SCENARIOS.get(lang_code, DISPLAY_SCENARIOS['en']).get(scenario_code, scenario_code)

    summary = {
        "Crop": crop_label,
        "Planting Date": planting_date.strftime('%Y-%m-%d'),
        "Harvest Date": harvest_date.strftime('%Y-%m-%d'),
        "Scenario": scenario_label,
        "Fertilizer (kg/ha)": management_inputs.get('fertilizer_kg_ha'),
        "Pesticide (L/ha)": management_inputs.get('pesticide_l_ha'),
        "Generated Rainfall (mm)": round(weather['total_precip_mm'], 1),
        "Avg Max Temp (°C)": round(weather['avg_temp_max_C'], 1),
        "Total Sunshine (hrs)": round(weather['total_sunshine_h'], 1)
    }

    return pred, summary, std

# ---------------- Main app ----------------
def main():
    st.set_page_config(page_title="Generative AI Farm Digital Twin", layout="wide")
    lang = st.sidebar.selectbox("Select Language / ಭಾಷೆ", ["English", "Kannada"])
    lang_code = 'en' if lang == "English" else 'kn'
    L = LANG_DICT[lang_code]

    st.title(L['title'])
    st.sidebar.header(L['sidebar_header'])

    models = load_models()
    if not models:
        st.error(L['models_not_loaded'])
        return

    # prepare localized dropdowns
    display_crops = [DISPLAY_CROPS[lang_code][c] for c in CROPS]
    crop_display_to_code = {DISPLAY_CROPS[lang_code][c]: c for c in CROPS}

    display_scenarios = [DISPLAY_SCENARIOS[lang_code][s] for s in SCENARIOS]
    scenario_display_to_code = {DISPLAY_SCENARIOS[lang_code][s]: s for s in SCENARIOS}

    # Sidebar inputs
    selected_crop_display = st.sidebar.selectbox(L['select_crop'], display_crops)
    crop = crop_display_to_code[selected_crop_display]  # internal code

    planting = st.sidebar.date_input(L['planting_date'], datetime.now().date())

    # Defaults
    default_fert = 150 if crop in ["Jowar", "Maize"] else 180
    default_pest = 10.0 if crop in ["Jowar", "Maize"] else 15.0

    fert = st.sidebar.slider(L['fertilizer_input'], min_value=0, max_value=500, value=default_fert, step=1)
    # pesticide slider 0.1..10.0 step 0.1
    pest = st.sidebar.slider(L['pesticide_input'], min_value=0.1, max_value=10.0, value=default_pest, step=0.1, format="%.1f")

    selected_scenario_display = st.sidebar.selectbox(L['scenario_selection'], display_scenarios)
    scenario_code = scenario_display_to_code[selected_scenario_display]  # internal code

    seed = st.sidebar.number_input(L['random_seed'], min_value=0, step=1, value=0)

    if st.sidebar.button(L['run_sim']):
        pred, summary, std = predict_yield_with_uncertainty(
            crop,
            datetime.combine(planting, datetime.min.time()),
            {"fertilizer_kg_ha": fert, "pesticide_l_ha": pest},
            scenario_code,
            models,
            seed if seed > 0 else None,
            lang_code=lang_code
        )
        if pred is None:
            st.error("Prediction failed — check model & inputs.")
            return
        st.session_state.pred = pred
        st.session_state.summary = summary
        st.session_state.std = std
        st.session_state.selected_crop_display = selected_crop_display
        st.session_state.selected_scenario_display = selected_scenario_display

    # Results area
    if "pred" in st.session_state:
        st.subheader(L['predicted_yield'])
        st.success(f"{st.session_state.pred:.2f} kg/ha")

        if st.session_state.std is not None:
            st.write(f"{L['uncertainty']}: {st.session_state.std:.2f}")

        # Validation & SHAP
        st.subheader(L['model_val'])
        metrics_file = f"models/validation_reports/{crop.lower()}_metrics.json"
        shap_img = f"models/validation_reports/{crop.lower()}_shap_summary.png"

        if os.path.exists(metrics_file):
            st.write(L['val_metrics'])
            try:
                st.json(json.load(open(metrics_file)))
            except Exception:
                st.write("Could not read metrics file.")
        if os.path.exists(shap_img):
            try:
                st.image(shap_img, caption=L['shap_summary'])
            except Exception:
                pass

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader(L['scenario_inputs'])
            display_table = st.session_state.summary.copy()
            # ensure Crop & Scenario remain localized
            display_table["Crop"] = st.session_state.selected_crop_display
            display_table["Scenario"] = st.session_state.selected_scenario_display
            st.table(pd.Series(display_table).to_frame('Value'))

        with c2:
            st.subheader(L['comparison'])
            # Compare base (NORMAL) and optimized
            base, _, _ = predict_yield_with_uncertainty(
                crop, datetime.combine(planting, datetime.min.time()),
                {"fertilizer_kg_ha": fert, "pesticide_l_ha": pest},
                'NORMAL', models, seed if seed > 0 else None, lang_code=lang_code
            )
            # Simple optimization step for "optimized" baseline: +20% fert, +50% pest
            opt_fert = fert * 1.2
            opt_pest = round(pest * 1.5, 1)
            opt, _, _ = predict_yield_with_uncertainty(
                crop, datetime.combine(planting, datetime.min.time()),
                {"fertilizer_kg_ha": opt_fert, "pesticide_l_ha": opt_pest},
                'NORMAL', models, seed if seed > 0 else None, lang_code=lang_code
            )
            df_compare = pd.DataFrame({
                "Scenario": [DISPLAY_SCENARIOS[lang_code]['NORMAL'], "Optimized"],
                "Predicted Yield (kg/ha)": [base, opt]
            })
            st.dataframe(df_compare)
            st.write(f"**{L['difference']}: {opt - base:.2f} kg/ha**")

        st.markdown("---")
        st.header(L['step10_header'])

        if st.button(L['find_optimal']):
            st.info(L['optimizing'])
            # Define search grid
            fert_range = list(range(50, 301, 10))            # 50..300 step 10
            pest_range = [round(x * 0.1, 1) for x in range(1, 101)]  # 0.1..10.0 step 0.1

            total = len(fert_range) * len(pest_range)
            prog = st.progress(0)
            best_y = -1.0
            best_params = (None, None)
            steps = 0

            # Grid search
            for f in fert_range:
                for p in pest_range:
                    y, _, _ = predict_yield_with_uncertainty(
                        crop,
                        datetime.combine(planting, datetime.min.time()),
                        {"fertilizer_kg_ha": f, "pesticide_l_ha": p},
                        scenario_code, models, seed if seed > 0 else None, lang_code=lang_code
                    )
                    # if model returns None (error), skip
                    if y is None:
                        steps += 1
                        prog.progress(steps / total)
                        continue
                    if y > best_y:
                        best_y = y
                        best_params = (f, p)
                    steps += 1
                    # update progress occasionally to reduce UI churn
                    if steps % 50 == 0 or steps == total:
                        prog.progress(steps / total)

            if best_params[0] is None:
                st.error("Optimization failed — no valid model predictions.")
            else:
                best_f, best_p = best_params
                st.success(f"{L['optimal_fert']}: {best_f} kg/ha")
                st.success(f"{L['optimal_pest']}: {best_p} L/ha")
                st.success(f"{L['predicted_yield_opt']}: {best_y:.2f} kg/ha")

                current, _, _ = predict_yield_with_uncertainty(
                    crop, datetime.combine(planting, datetime.min.time()),
                    {"fertilizer_kg_ha": fert, "pesticide_l_ha": pest},
                    scenario_code, models, seed if seed > 0 else None, lang_code=lang_code
                )
                st.info(f"{L['improvement']}: {best_y - current:.2f} kg/ha")

if __name__ == "__main__":
    main()
