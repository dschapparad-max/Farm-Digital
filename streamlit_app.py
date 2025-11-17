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

# --------------------- Language dictionaries ---------------------
LANG_DICT = {
    'en': {
        'title': "🌾 Generative AI-Powered Farm Digital Twin",
        'sidebar_header': "🚜 Management Inputs & Scenario Setup",
        'select_crop': "1. Select Crop:",
        'planting_date': "2. Select Planting Date:",
        'fertilizer_input': "3. Fertilizer Input (kg/ha):",
        'irrigation_input': "4. Irrigation Input (m³/ha):",
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
        'find_optimal': "Find Optimal Fertilizer & Irrigation Inputs",
        'optimal_fert': "Optimal Fertilizer",
        'optimal_irr': "Optimal Irrigation",
        'predicted_yield_opt': "Predicted Yield with Optimized Inputs",
        'improvement': "Improvement over current inputs",
        'models_not_loaded': "Models not loaded. Please ensure model files exist in 'models/' directory."
    },
    'kn': {
        'title': "🌾 ಜನರೇಟಿವ್ AI ಚಾಲಿತ ಕೃಷಿ ಡಿಜಿಟಲ್ ಟ್ವಿನ್",
        'sidebar_header': "🚜 ನಿರ್ವಹಣಾ ಇನ್ಪುಟ್‌ಗಳು ಮತ್ತು ಹವಾಮಾನ ಸ್ಥಿತಿಗತಿ",
        'select_crop': "1. ಬೆಳೆ ಆಯ್ಕೆ ಮಾಡಿ:",
        'planting_date': "2. ಬಿತ್ತನೆ ದಿನಾಂಕ ಆಯ್ಕೆ ಮಾಡಿ:",
        'fertilizer_input': "3. ರಾಸಾಯನಿಕ ಪೋಷಕಾಂಶ (ಕೆಜಿ/ಹೆಕ್ಟೇರ್):",
        'irrigation_input': "4. ಸಿಂಚನ (ಮೀಟರ್³/ಹೆಕ್ಟೇರ್):",
        'scenario_selection': "5. ಭವಿಷ್ಯದ ಹವಾಮಾನ ಸಂದರ್ಭ ಆಯ್ಕೆ ಮಾಡಿ:",
        'random_seed': "ಯಾದೃಚ್ಛಿಕ ಬೀಜ (0 = ಯಾದೃಚ್ಛಿಕ)",
        'run_sim': "ಡಿಜಿಟಲ್ ಟ್ವಿನ್ ಸಿಮ್ಯುಲೇಷನ್ ಚಾಲನೆ ಮಾಡಿ",
        'predicted_yield': "ಮುನ್ಸೂಚನೆ ಮಾಡಿದ ಉತ್ಪಾದನೆ",
        'uncertainty': "ಅನುಮಾನ (ವೃಕ್ಷಗಳಲ್ಲಿನ ಸಣ್ಣ ಪ್ರಮಾಣ)",
        'model_val': "ಮಾದರಿ ಪರಿಶೀಲನೆ ಮತ್ತು ವಿವರಣೆ",
        'val_metrics': "ಪರಿಶೀಲನೆ ಮೆಟ್ರಿಕ್ಸ್ (ಪರೀಕ್ಷಾ ಸೆಟ್)",
        'shap_summary': "SHAP ಸಾರಾಂಶ (ವಿಶ್ವ ವೈಶಿಷ್ಟ್ಯ ಮಹತ್ವ)",
        'scenario_inputs': "ಹವಾಮಾನ ಪರಿಸ್ಥಿತಿಯಲ್ಲಿ ಉಪಯೋಗಿಸಿದ ಇನ್ಪುಟ್‌ಗಳು",
        'comparison': "ನಿರ್ವಹಣೆ 'ಎಲ್ಲಿ' ಹೋಲಿಕೆ (ಸಾಮಾನ್ಯದ ವಿರುದ್ಧ)",
        'difference': "ವ್ಯತ್ಯಾಸ (ಶ್ರೇಷ್ಠೀಕೃತ vs ಮೂಲ)",
        'step10_header': "🔧 کریಷಿಕರಿಗಾಗಿ ಪರಿಪೂರ್ಣತೆ",
        'find_optimal': "ಪರಿಪೂರ್ಣ ರಾಸಾಯನಿಕ ಮತ್ತು ಸಿಂಚನ ಇನ್ಪುಟ್‌ಗಳನ್ನು ಹುಡುಕಿ",
        'optimal_fert': "ಪರಿಪೂರ್ಣ ರಾಸಾಯನಿಕ ಪೋಷಕಾಂಶ",
        'optimal_irr': "ಪರಿಪೂರ್ಣ ಸಿಂಚನ ಪ್ರಮಾಣ",
        'predicted_yield_opt': "ಪರಿಪೂರ್ಣ ಇನ್ಪುಟ್‌ಗಳೊಂದಿಗೆ ಮುನ್ಸೂಚನೆ ಮಾಡಿದ ಉತ್ಪಾದನೆ",
        'improvement': "ಪ್ರಸ್ತುತ ಇನ್ಪುಟ್‌ಗಳಿಗಿಂತ ಸುಧಾರಣೆ",
        'models_not_loaded': "ಮಾದರಿ ಲೋಡ್ ಆಗುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು 'models/' ಡೈರಕ್ಟರಿಯಲ್ಲಿ ಮಾದರಿ ಫೈಲ್‌ಗಳನ್ನು ಖಚಿತಗೊಳಿಸಿ."
    }
}
# --------------------- End Language dictionaries ---------------------

@st.cache_resource
def load_models():
    models = {}
    for crop in CROPS:
        model_path = os.path.join(MODELS_PATH, f'{crop.lower()}_model.joblib')
        if os.path.exists(model_path):
            models[crop] = joblib.load(model_path)
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
    return {
        'total_precip_mm': max(0, base_precip * precip_factor + float(rng.uniform(-50,50))),
        'avg_temp_max_C': float(base_temp_max + temp_max_adjustment + float(rng.uniform(-1,1))),
        'total_sunshine_h': max(0, base_sunshine * (1 + float(rng.uniform(-0.1,0.1))))
    }


def predict_yield_with_uncertainty(crop, planting_date, management_inputs, scenario_type, models, seed=None):
    if crop not in models:
        return None, None, None
    
    model = models[crop]
    duration = GROWING_PERIODS.get(crop, 150)
    harvest_date = planting_date + timedelta(days=duration)

    weather = generate_future_weather_scenario(duration, scenario_type, seed)
    X = pd.DataFrame([{
        'fertilizer_kg_ha': management_inputs['fertilizer_kg_ha'],
        'irrigation_m3_ha': management_inputs['irrigation_m3_ha'],
        **weather
    }])

    pred = float(model.predict(X)[0])
    pred = max(100, pred)

    std = None
    if hasattr(model, "estimators_"):
        std = float(np.std([t.predict(X)[0] for t in model.estimators_]))

    summary = {
        "Planting Date": planting_date.strftime('%Y-%m-%d'),
        "Harvest Date": harvest_date.strftime('%Y-%m-%d'),
        "Scenario": scenario_type,
        "Fertilizer (kg/ha)": management_inputs['fertilizer_kg_ha'],
        "Irrigation (m³/ha)": management_inputs['irrigation_m3_ha'],
        "Generated Rainfall (mm)": f"{weather['total_precip_mm']:.0f}",
        "Avg Max Temp (°C)": f"{weather['avg_temp_max_C']:.1f}",
        "Total Sunshine (hrs)": f"{weather['total_sunshine_h']:.0f}"
    }

    return pred, summary, std


def main():
    st.set_page_config(page_title="Generative AI Farm Digital Twin", layout="wide")

    lang = st.sidebar.selectbox("Select Language / ಭಾಷೆ", ["English", "Kannada"])
    L = LANG_DICT['en'] if lang == "English" else LANG_DICT['kn']

    st.title(L['title'])

    models = load_models()
    if not models:
        st.error(L['models_not_loaded'])
        return

    st.sidebar.header(L['sidebar_header'])
    crop = st.sidebar.selectbox(L['select_crop'], CROPS)
    planting = st.sidebar.date_input(L['planting_date'], datetime(2025, 4, 1))

    default_fert = 150 if crop in ["Jowar", "Maize"] else 180
    default_irr = 500 if crop in ["Jowar", "Maize"] else 700

    fert = st.sidebar.slider(L['fertilizer_input'], 50, 300, default_fert)
    irr = st.sidebar.slider(L['irrigation_input'], 100, 2000, default_irr)
    scenario = st.sidebar.selectbox(L['scenario_selection'], SCENARIOS)
    seed = st.sidebar.number_input(L['random_seed'], 0)

    if st.sidebar.button(L['run_sim']):
        pred, summary, std = predict_yield_with_uncertainty(
            crop, datetime.combine(planting, datetime.min.time()),
            {"fertilizer_kg_ha": fert, "irrigation_m3_ha": irr},
            scenario, models, seed if seed > 0 else None
        )
        st.session_state.pred = pred
        st.session_state.summary = summary
        st.session_state.std = std

    if "pred" in st.session_state:
        st.subheader(L['predicted_yield'])
        st.success(f"{st.session_state.pred:.2f} kg/ha")

        if st.session_state.std:
            st.write(f"{L['uncertainty']}: {st.session_state.std:.2f}")

        # ---------------- VALIDATION BLOCK (sample preview REMOVED) ----------------
        st.subheader(L['model_val'])

        metrics_file = f"models/validation_reports/{crop.lower()}_metrics.json"
        shap_img = f"models/validation_reports/{crop.lower()}_shap_summary.png"

        if os.path.exists(metrics_file):
            st.write(L['val_metrics'])
            st.json(json.load(open(metrics_file)))

        if os.path.exists(shap_img):
            st.image(shap_img, caption=L['shap_summary'])

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader(L['scenario_inputs'])
            st.table(pd.Series(st.session_state.summary).to_frame('Value'))

        with c2:
            st.subheader(L['comparison'])
            base, _, _ = predict_yield_with_uncertainty(
                crop, datetime.combine(planting, datetime.min.time()),
                {"fertilizer_kg_ha": fert, "irrigation_m3_ha": irr},
                "NORMAL", models, seed if seed > 0 else None
            )
            opt_fert = fert * 1.2
            opt_irr = irr * 1.5
            opt, _, _ = predict_yield_with_uncertainty(
                crop, datetime.combine(planting, datetime.min.time()),
                {"fertilizer_kg_ha": opt_fert, "irrigation_m3_ha": opt_irr},
                "NORMAL", models, seed if seed > 0 else None
            )
            st.dataframe(pd.DataFrame({
                "Scenario": ["Base (Normal)", "Optimized"],
                "Predicted Yield (kg/ha)": [base, opt]
            }))
            st.write(f"**{L['difference']}: {opt - base:.2f} kg/ha**")

        st.markdown("---")
        st.header(L['step10_header'])

        if st.button(L['find_optimal']):
            best_y = -1
            best_f, best_i = None, None

            prog = st.progress(0)
            steps = 0
            total = 26 * 20

            for f in range(50, 301, 10):
                for i in range(100, 2001, 100):
                    y, _, _ = predict_yield_with_uncertainty(
                        crop, datetime.combine(planting, datetime.min.time()),
                        {"fertilizer_kg_ha": f, "irrigation_m3_ha": i},
                        scenario, models, seed if seed > 0 else None
                    )
                    if y > best_y:
                        best_y, best_f, best_i = y, f, i
                    steps += 1
                    prog.progress(steps / total)

            st.success(f"{L['optimal_fert']}: {best_f} kg/ha")
            st.success(f"{L['optimal_irr']}: {best_i} m³/ha")
            st.success(f"{L['predicted_yield_opt']}: {best_y:.2f} kg/ha")

            current, _, _ = predict_yield_with_uncertainty(
                crop, datetime.combine(planting, datetime.min.time()),
                {"fertilizer_kg_ha": fert, "irrigation_m3_ha": irr},
                scenario, models, seed if seed > 0 else None
            )
            st.info(f"{L['improvement']}: {best_y - current:.2f} kg/ha")


if __name__ == "__main__":
    main()
