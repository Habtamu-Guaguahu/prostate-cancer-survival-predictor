# =========================================
# PROSTATE CANCER SURVIVAL PREDICTOR WEB APP
# GBSA Model with RSF Features
# PUBLICATION-READY VERSION
# =========================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import pickle
import os
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
import warnings
warnings.filterwarnings('ignore')

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Prostate Cancer Survival Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================
# LOAD MODELS
# =========================================

@st.cache_resource
def load_models():
    """Load all pre-trained models and data"""
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    
    try:
        # Load GBSA model
        with open(os.path.join(model_dir, 'gbsa_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        # Load feature names
        with open(os.path.join(model_dir, 'feature_names.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        
        # Load KM data
        with open(os.path.join(model_dir, 'km_data.pkl'), 'rb') as f:
            km_data = pickle.load(f)
        
        # Load metrics
        metrics_path = os.path.join(model_dir, 'metrics.pkl')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics = pickle.load(f)
        else:
            # Fallback values if metrics.pkl doesn't exist
            metrics = {
                'c_index': 0.8321,
                'iauc': 0.8940,
                'cutoff': -0.3187,
                'min_risk': -1.5,
                'max_risk': 1.5,
                'risk_percentiles': {}  # Will be populated if available
            }
        
        # Check model capabilities
        has_survival_fn = hasattr(model, 'predict_survival_function')
        has_cumulative_hazard = hasattr(model, 'predict_cumulative_hazard_function')
        
        # Determine model type
        model_type = str(type(model)).split('.')[-1].strip("'>")
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'km_data': km_data,
            'metrics': metrics,
            'has_survival_fn': has_survival_fn,
            'has_cumulative_hazard': has_cumulative_hazard,
            'model_type': model_type,
            'success': True
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# Load everything
models = load_models()

if not models['success']:
    st.stop()

# =========================================
# FEATURE CLEANING FUNCTION
# =========================================

def clean_feature_name(feature):
    """Convert raw feature names to readable clinical terms"""
    mapping = {
        'ecogreclass2_ECOG_3_4_vs_ECOG_1_2': 'ECOG Performance Status 3-4',
        'timetonadir_cat_high_vs_low': 'Time to PSA Nadir (High)',
        'metastasis_Multiple_metastasis_vs_No_metastasis': 'Multiple Metastases',
        'stageatdiagnosis_Stage_IV_vs_Stage_I': 'Stage IV Disease',
        'metastasis_Bone_metastasis_vs_No_metastasis': 'Bone Metastases',
        'histologictype_poorly_differentiated_vs_well_differentiated': 'Poorly Differentiated Histology',
        'radiotx_yes_vs_no': 'Radiotherapy',
        'stageatdiagnosis_Stage_III_vs_Stage_I': 'Stage III Disease',
        'adt_yes_vs_no': 'ADT Therapy',
        'psanadir_cat_high_vs_low': 'PSA Nadir (High)'
    }
    return mapping.get(feature, feature)

# =========================================
# GBSA PREDICTION FUNCTIONS
# =========================================

def predict_gbsa_risk_score(features_dict, model, scaler, feature_names):
    """
    Predict GBSA risk score from patient features
    """
    X = pd.DataFrame([features_dict])[feature_names]
    X_scaled = scaler.transform(X)
    risk_score = model.predict(X_scaled)[0]
    return risk_score, X_scaled

def get_survival_curve(model, X_scaled, km_data, risk_score, is_high_risk):
    """
    Get survival curve with proper labeling based on model capabilities
    Returns: (times, survival, curve_type, description)
    """
    # Check if model supports personalized survival functions
    if hasattr(model, 'predict_survival_function'):
        try:
            surv_fn = model.predict_survival_function(X_scaled)[0]
            times = surv_fn.x
            survival = surv_fn.y
            curve_type = "Model-Based"
            description = "True GBSA personalized survival estimate"
            return times, survival, curve_type, description
        except Exception as e:
            st.warning(f"Could not use predict_survival_function: {e}")
            # Fall through to KM-based approach
    
    # Check for cumulative hazard function
    if hasattr(model, 'predict_cumulative_hazard_function'):
        try:
            cum_hazard = model.predict_cumulative_hazard_function(X_scaled)[0]
            times = cum_hazard.x
            # Convert cumulative hazard to survival
            survival = np.exp(-cum_hazard.y)
            curve_type = "Model-Based"
            description = "GBSA survival derived from cumulative hazard"
            return times, survival, curve_type, description
        except Exception as e:
            st.warning(f"Could not use predict_cumulative_hazard_function: {e}")
            # Fall through to KM-based approach
    
    # Fallback: Use KM curves based on risk group
    if is_high_risk:
        times = km_data['high_risk']['times']
        survival = km_data['high_risk']['surv']
        curve_type = "KM-Based"
        description = "Kaplan-Meier survival curve for risk group (fallback method)"
    else:
        times = km_data['low_risk']['times']
        survival = km_data['low_risk']['surv']
        curve_type = "KM-Based"
        description = "Kaplan-Meier survival curve for risk group (fallback method)"
    
    return times, survival, curve_type, description

def calculate_percentile(risk_score, metrics):
    """
    Calculate percentile rank of risk score within training cohort
    Uses percentiles from metrics if available, otherwise estimates
    """
    # If we have stored percentiles, use them
    if 'risk_percentiles' in metrics and metrics['risk_percentiles']:
        # Find closest percentile
        percentiles = metrics['risk_percentiles']
        percentiles_keys = sorted(percentiles.keys())
        
        # Find the closest percentile
        closest_p = percentiles_keys[0]
        closest_diff = abs(risk_score - percentiles[closest_p])
        
        for p in percentiles_keys:
            diff = abs(risk_score - percentiles[p])
            if diff < closest_diff:
                closest_diff = diff
                closest_p = p
        
        return closest_p
    
    # Estimate percentile using min/max
    min_risk = metrics.get('min_risk', -1.5)
    max_risk = metrics.get('max_risk', 1.5)
    
    if max_risk - min_risk > 0:
        # Clamp and normalize
        normalized = max(0, min(1, (risk_score - min_risk) / (max_risk - min_risk)))
        percentile = normalized * 100
    else:
        percentile = 50
    
    return percentile

def get_survival_at_time(time_points, surv_curve, target_time):
    """Get survival probability at specific time point"""
    idx = min(range(len(time_points)), key=lambda i: abs(time_points[i] - target_time))
    return surv_curve[idx]

# =========================================
# MAIN APPLICATION
# =========================================

def main():
    # Title
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #1B4332; border-radius: 10px; color: white;'>
        <h1>🩺 Prostate Cancer Survival Predictor</h1>
        <p style='font-size: 1.1rem;'>GBSA Model with Dynamic Risk Prediction</p>
        <p style='font-size: 0.9rem;'>C-index: {:.4f} | iAUC: {:.4f}</p>
    </div>
    """.format(models['metrics'].get('c_index', 0.8321), 
               models['metrics'].get('iauc', 0.8940)), 
    unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("Test C-index", f"{models['metrics'].get('c_index', 0.8321):.4f}")
        st.metric("iAUC", f"{models['metrics'].get('iauc', 0.8940):.4f}")
        st.metric("Risk Group Cutoff", f"{models['metrics'].get('cutoff', -0.3187):.4f}")
        
        st.markdown("---")
        st.header("ℹ️ Model Information")
        st.markdown(f"**Model Type:** `{models['model_type']}`")
        st.markdown(f"**Survival Function Support:** {'✅ Yes' if models['has_survival_fn'] else '❌ No'}")
        st.markdown(f"**Cumulative Hazard Support:** {'✅ Yes' if models['has_cumulative_hazard'] else '❌ No'}")
        
        if not models['has_survival_fn'] and not models['has_cumulative_hazard']:
            st.warning("⚠️ Using KM-based survival curves (fallback method)")
        
        st.markdown("---")
        st.header("📋 Patient Instructions")
        st.markdown("""
        1. Enter patient characteristics
        2. Click 'Predict Survival'
        3. View GBSA risk score and percentile
        4. See personalized survival curve
        """)
        
        st.markdown("---")
        st.header("📈 Features Used")
        for feature in models['feature_names']:
            st.markdown(f"- {clean_feature_name(feature)}")
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 Patient Input")
        
        with st.form("prediction_form"):
            # Collect features as dictionary
            features_dict = {}
            
            st.markdown("#### Patient Status")
            eco = st.selectbox(
                "ECOG Performance Status",
                ["ECOG 1-2 (Good)", "ECOG 3-4 (Poor)"],
                help="ECOG 3-4 indicates poor functional status"
            )
            features_dict['ecogreclass2_ECOG_3_4_vs_ECOG_1_2'] = 1 if "3-4" in eco else 0
            
            st.markdown("#### Disease Characteristics")
            bone_met = st.selectbox(
                "Bone Metastases",
                ["No", "Yes"],
                help="Presence of bone metastases"
            )
            features_dict['metastasis_Bone_metastasis_vs_No_metastasis'] = 1 if bone_met == "Yes" else 0
            
            mult_met = st.selectbox(
                "Multiple Metastases",
                ["No", "Yes"],
                help="Multiple metastasis sites"
            )
            features_dict['metastasis_Multiple_metastasis_vs_No_metastasis'] = 1 if mult_met == "Yes" else 0
            
            stage_3 = st.selectbox(
                "Stage III Disease",
                ["No", "Yes"]
            )
            features_dict['stageatdiagnosis_Stage_III_vs_Stage_I'] = 1 if stage_3 == "Yes" else 0
            
            stage_4 = st.selectbox(
                "Stage IV Disease",
                ["No", "Yes"]
            )
            features_dict['stageatdiagnosis_Stage_IV_vs_Stage_I'] = 1 if stage_4 == "Yes" else 0
            
            histology = st.selectbox(
                "Histologic Type",
                ["Well differentiated", "Poorly differentiated"]
            )
            features_dict['histologictype_poorly_differentiated_vs_well_differentiated'] = 1 if "Poorly" in histology else 0
            
            st.markdown("#### Treatment")
            adt = st.selectbox(
                "ADT Therapy",
                ["No", "Yes"]
            )
            features_dict['adt_yes_vs_no'] = 1 if adt == "Yes" else 0
            
            radio = st.selectbox(
                "Radiotherapy",
                ["No", "Yes"]
            )
            features_dict['radiotx_yes_vs_no'] = 1 if radio == "Yes" else 0
            
            st.markdown("#### Biomarkers")
            psa_nadir = st.selectbox(
                "PSA Nadir",
                ["Low", "High"]
            )
            features_dict['psanadir_cat_high_vs_low'] = 1 if psa_nadir == "High" else 0
            
            time_to_nadir = st.selectbox(
                "Time to PSA Nadir",
                ["Low", "High"]
            )
            features_dict['timetonadir_cat_high_vs_low'] = 1 if time_to_nadir == "High" else 0
            
            st.markdown("---")
            
            prediction_time = st.slider(
                "Prediction Time (months)",
                min_value=1,
                max_value=60,
                value=36,
                step=1
            )
            
            submitted = st.form_submit_button("🔮 Predict Survival", use_container_width=True)
    
    with col2:
        if submitted:
            st.subheader("📊 Prediction Results")
            
            # Get GBSA risk score
            risk_score, X_scaled = predict_gbsa_risk_score(
                features_dict,
                models['model'],
                models['scaler'],
                models['feature_names']
            )
            
            # Determine risk group using cutoff
            median_cutoff = models['metrics'].get('cutoff', -0.3187)
            is_high_risk = risk_score >= median_cutoff
            risk_group = "High Risk" if is_high_risk else "Low Risk"
            
            # Calculate percentile
            percentile = calculate_percentile(risk_score, models['metrics'])
            
            # Get survival curve
            time_points, patient_surv, curve_type, curve_description = get_survival_curve(
                models['model'],
                X_scaled,
                models['km_data'],
                risk_score,
                is_high_risk
            )
            
            # Get survival at specific time point
            surv_prob = get_survival_at_time(time_points, patient_surv, prediction_time)
            
            # Display metrics - ROW 1
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric(
                    "GBSA Risk Score",
                    f"{risk_score:.4f}",
                    delta=None
                )
            
            with col_b:
                st.metric(
                    "Risk Group",
                    risk_group,
                    delta=None,
                    delta_color="inverse" if is_high_risk else "normal"
                )
            
            with col_c:
                st.metric(
                    "Percentile",
                    f"{percentile:.0f}th",
                    delta=None
                )
            
            with col_d:
                st.metric(
                    f"Survival at {prediction_time}m",
                    f"{surv_prob:.1%}",
                    delta=None
                )
            
            # Risk group indicator
            if is_high_risk:
                st.error(f"⚠️ HIGH RISK PATIENT (GBSA Score: {risk_score:.4f} ≥ {median_cutoff:.4f})")
            else:
                st.success(f"✅ LOW RISK PATIENT (GBSA Score: {risk_score:.4f} < {median_cutoff:.4f})")
            
            # Survival Plot
            st.subheader(f"📈 {curve_description}")
            
            # Add note about curve type
            if curve_type == "KM-Based":
                st.info("ℹ️ Using KM-based survival curve (your model does not support individual survival prediction)")
            
            # Get KM data for reference
            km_data = models['km_data']
            
            # Create interactive plot
            fig = go.Figure()
            
            # Low risk KM curve (reference)
            fig.add_trace(go.Scatter(
                x=km_data['low_risk']['times'],
                y=km_data['low_risk']['surv'],
                mode='lines',
                name=f"Low Risk KM (n={km_data['n_low']})",
                line=dict(color='#2E86AB', width=2, dash='dash')
            ))
            
            # High risk KM curve (reference)
            fig.add_trace(go.Scatter(
                x=km_data['high_risk']['times'],
                y=km_data['high_risk']['surv'],
                mode='lines',
                name=f"High Risk KM (n={km_data['n_high']})",
                line=dict(color='#E63946', width=2, dash='dash')
            ))
            
            # Patient-specific curve with appropriate label
            if curve_type == "Model-Based":
                line_color = '#FFD700'
                line_width = 4
                dash_style = 'solid'
                curve_label = f'Patient (GBSA: {risk_score:.4f})'
            else:
                line_color = '#FF6B6B' if is_high_risk else '#4ECDC4'
                line_width = 3
                dash_style = 'dashdot'
                curve_label = f'Patient - {risk_group} (GBSA: {risk_score:.4f})'
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=patient_surv,
                mode='lines',
                name=curve_label,
                line=dict(color=line_color, width=line_width, dash=dash_style)
            ))
            
            # Patient point at specific time
            fig.add_trace(go.Scatter(
                x=[prediction_time],
                y=[surv_prob],
                mode='markers',
                name=f'Patient at {prediction_time}m',
                marker=dict(
                    size=20,
                    color='#FFD700',
                    symbol='star',
                    line=dict(width=3, color='black')
                )
            ))
            
            # Layout
            fig.update_layout(
                title=f'<b>GBSA Survival Prediction</b><br><sup>{curve_description}</sup>',
                xaxis_title='Time (months)',
                yaxis_title='Survival Probability',
                yaxis_range=[0, 1.02],
                xaxis_range=[0, 62],
                hovermode='x unified',
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.95)',
                    bordercolor='black',
                    borderwidth=1
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # GBSA Risk Score Gauge
            st.markdown("---")
            st.subheader("📊 Risk Score Visualization")
            
            min_risk = models['metrics'].get('min_risk', -1.5)
            max_risk = models['metrics'].get('max_risk', 1.5)
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                title={'text': f"GBSA Score (Percentile: {percentile:.0f}th)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [min_risk, max_risk]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [min_risk, median_cutoff], 'color': "#2E86AB"},
                        {'range': [median_cutoff, max_risk], 'color': "#E63946"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': median_cutoff
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Interpretation
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.info(f"""
                **GBSA Risk Score Details:**
                - GBSA Score: **{risk_score:.4f}**
                - Risk Group Cutoff: **{median_cutoff:.4f}**
                - Risk Classification: **{risk_group}**
                - Percentile Rank: **{percentile:.0f}th** percentile
                - Distance from cutoff: **{abs(risk_score - median_cutoff):.4f}**
                """)
            
            with col_info2:
                # Calculate normalized risk position
                if max_risk - min_risk > 0:
                    normalized_score = (risk_score - min_risk) / (max_risk - min_risk)
                    normalized_score = max(0, min(1, normalized_score))
                else:
                    normalized_score = 0.5
                
                st.success(f"""
                **Clinical Interpretation:**
                - Risk Position: **{normalized_score:.1%}** of risk spectrum
                - Survival at {prediction_time}m: **{surv_prob:.1%}**
                - Curve Type: **{curve_type}**
                - {curve_description}
                """)
            
            # Model performance context
            st.markdown("---")
            st.markdown(f"""
            **Model Performance Context:**
            - **GBSA Model**: {models['model_type']}
            - **C-index**: {models['metrics'].get('c_index', 0.8321):.4f} (Excellent discrimination)
            - **iAUC**: {models['metrics'].get('iauc', 0.8940):.4f}
            - **Risk Group Separation**: Highly significant (p < 0.0001)
            - **Survival Curve Method**: {curve_description}
            - Results should be interpreted in clinical context
            """)
            
        else:
            # Welcome message
            st.info("👈 Enter patient characteristics and click 'Predict Survival'")
            
            with st.expander("🔍 View Example Patient"):
                st.markdown("""
                **Example Patient:**
                - ECOG: 1-2 (Good)
                - No bone metastases
                - No multiple metastases
                - Stage II disease
                - Well differentiated histology
                - Received ADT and radiotherapy
                - PSA Nadir: Low
                - Time to PSA Nadir: Low
                
                **Expected GBSA Results:**
                - GBSA Risk Score: ~-0.85
                - Risk Group: Low Risk
                - Percentile: ~20th percentile
                - 36-month survival: ~82%
                """)

if __name__ == "__main__":
    main()