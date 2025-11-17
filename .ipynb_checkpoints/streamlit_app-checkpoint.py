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
        'validation_sample': "Sample of validation predictions (first 10 rows)",
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
        'validation_sample': "ಪರಿಶೀಲನೆ ಮುನ್ಸೂಚನೆ ಮಾದರಿಗಳ ಮಾದರಿ (ಮೊದಲ 10 ಸಾಲುಗಳು)",
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
            try:
                models[crop] = joblib.load(model_path)
            except Exception as e:
                st.error(f"Error loading model for {crop}: {e}")
        else:
            # Don't error loudly for every crop during load; return empty dict and show later
            pass
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

    # ----------------- Language Selection -----------------
    lang = st.sidebar.selectbox("Select Language / ಭಾಷೆ ಆಯ್ಕೆಮಾಡಿ", options=['English', 'Kannada'])
    lang_code = 'en' if lang == 'English' else 'kn'
    L = LANG_DICT[lang_code]
    # ----------------- End Language Selection -----------------

    st.title(L['title'])

    models = load_models()
    if not models:
        st.error(L['models_not_loaded'])
        return

    st.sidebar.header(L['sidebar_header'])
    
    selected_crop = st.sidebar.selectbox(L['select_crop'], CROPS)
    planting_date = st.sidebar.date_input(L['planting_date'], value=datetime(2025,4,1), min_value=datetime(2025,1,1), max_value=datetime(2025,12,31))

    if selected_crop in ['Jowar','Maize']:
        default_fert = 150
        default_irr = 500
    else:
        default_fert = 180
        default_irr = 700

    fertilizer_input = st.sidebar.slider(L['fertilizer_input'], min_value=50, max_value=300, value=default_fert, step=5)
    irrigation_input = st.sidebar.slider(L['irrigation_input'], min_value=100, max_value=2000, value=default_irr, step=50)
    scenario_selection = st.sidebar.selectbox(L['scenario_selection'], SCENARIOS)
    seed_input = st.sidebar.number_input(L['random_seed'], value=0, min_value=0, step=1)
    seed_val = int(seed_input) if seed_input > 0 else None

    management_inputs = {'fertilizer_kg_ha': fertilizer_input, 'irrigation_m3_ha': irrigation_input}

    st.header(f"Results for: {selected_crop}")

    # Initialize session state
    if "simulation_ran" not in st.session_state:
        st.session_state.simulation_ran = False
        st.session_state.predicted_yield = None
        st.session_state.summary_data = None
        st.session_state.uncertainty = None
        st.session_state.optimization_done = False
        st.session_state.best_yield = None
        st.session_state.best_fert = None
        st.session_state.best_irr = None

    # Run Simulation button (sidebar, to match first code)
    if st.sidebar.button(L['run_sim']):
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
        else:
            st.session_state.simulation_ran = True
            st.session_state.predicted_yield = predicted_yield
            st.session_state.summary_data = summary_data
            st.session_state.uncertainty = uncertainty
            # reset optimization flag so user can re-run optimization if desired
            st.session_state.optimization_done = False
            st.session_state.best_yield = None
            st.session_state.best_fert = None
            st.session_state.best_irr = None

    # Display results if simulation ran
    if st.session_state.simulation_ran:
        st.markdown(f"## {L['predicted_yield']}")
        st.success(f"**{st.session_state.predicted_yield:.2f} kg/ha**", icon="📈")
        if st.session_state.uncertainty is not None:
            st.write(f"{L['uncertainty']}: {st.session_state.uncertainty:.2f} kg/ha")

        # ------------------ Model Validation & Explainability ------------------
        st.markdown("### " + L['model_val'])

        # Primary expected files
        metrics_file = f"models/validation_reports/{selected_crop.lower()}_metrics.json"
        shap_img = f"models/validation_reports/{selected_crop.lower()}_shap_summary.png"
        val_csv = f"models/validation_reports/{selected_crop.lower()}_validation.csv"
        # Fallback metadata path
        alt_meta = os.path.join(MODELS_PATH, f"{selected_crop.lower()}_model.joblib.meta.json")

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
            st.write("**" + L['val_metrics'] + "**")
            st.json(metrics)
        else:
            st.info("Validation metrics not found. Run training to generate them.")

        # Show SHAP image if available
        if os.path.exists(shap_img):
            st.image(shap_img, caption=L['shap_summary'])
        else:
            st.info(L['shap_summary'] + " not available.")

        # Validation CSV preview
        if os.path.exists(val_csv):
            try:
                df_val = pd.read_csv(val_csv)
                st.write(L['validation_sample'])
                st.dataframe(df_val.head(10))
            except Exception as e:
                st.warning(f"Could not read validation CSV: {e}")
        else:
            st.info(L['validation_sample'] + " not found.")
        # -----------------------------------------------------------------------

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(L['scenario_inputs'])
            st.table(pd.Series(st.session_state.summary_data).to_frame('Value'))
        with col2:
            st.subheader(L['comparison'])
            yield_base, _, _ = predict_yield_with_uncertainty(selected_crop, datetime.combine(planting_date, datetime.min.time()), management_inputs, 'NORMAL', models, seed=seed_val)
            optimized_inputs = {'fertilizer_kg_ha': fertilizer_input * 1.2, 'irrigation_m3_ha': irrigation_input * 1.5}
            yield_optimized, _, _ = predict_yield_with_uncertainty(selected_crop, datetime.combine(planting_date, datetime.min.time()), optimized_inputs, 'NORMAL', models, seed=seed_val)
            comparison_data = {"Scenario": ["Base (Normal Weather)", "Optimized (Normal Weather)"], "Predicted Yield (kg/ha)": [f"{yield_base:.2f}", f"{yield_optimized:.2f}"]}
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, hide_index=True)
            st.markdown(f"**{L['difference']}:** **{yield_optimized - yield_base:.2f} kg/ha**")

        st.markdown("---")
        st.header(L['step10_header'])

        # Button to run optimization (manual) - same behavior as first code
        if st.button(L['find_optimal']):
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

            st.success(f"{L['optimal_fert']}: {best_fert} kg/ha")
            st.success(f"{L['optimal_irr']}: {best_irr} m³/ha")
            st.success(f"{L['predicted_yield_opt']}: {best_yield:.2f} kg/ha")

            # Compare with current inputs
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
                st.info(f"{L['improvement']}: {diff:.2f} kg/ha")

    # end if simulation_ran

if __name__ == "__main__":
    main()
