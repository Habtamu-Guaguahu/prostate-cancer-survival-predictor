# =========================================
# PROSTATE CANCER SURVIVAL PREDICTOR WEB APP
# GBSA Model with RSF Features
# Test C-index: 0.8321
# DYNAMIC RISK SCORE VERSION
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
from scipy.interpolate import interp1d
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
        
        return {
            'model': model,
            'scaler': scaler,
            'feature_names': feature_names,
            'km_data': km_data,
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

def get_feature_options(feature):
    """Get display options for each feature"""
    if 'ecog' in feature:
        return ['No (ECOG 1-2)', 'Yes (ECOG 3-4)']
    elif 'metastasis' in feature:
        return ['No', 'Yes']
    elif 'timetonadir' in feature:
        return ['Low', 'High']
    elif 'stageatdiagnosis' in feature:
        return ['No', 'Yes']
    elif 'histologictype' in feature:
        return ['No (Well diff)', 'Yes (Poorly diff)']
    elif 'radiotx' in feature or 'adt' in feature:
        return ['No', 'Yes']
    elif 'psanadir' in feature:
        return ['Low', 'High']
    else:
        return ['No', 'Yes']

# =========================================
# DYNAMIC PREDICTION FUNCTIONS
# =========================================

def predict_risk_score(features, model, scaler):
    """Predict risk score from patient features"""
    feature_array = np.array(features).reshape(1, -1)
    feature_scaled = scaler.transform(feature_array)
    risk_score = model.predict(feature_scaled)[0]
    return risk_score

def get_patient_survival_curve(risk_score, km_data, time_points=None):
    """
    Generate patient-specific survival curve based on risk score
    Using interpolation between low and high risk curves
    """
    if time_points is None:
        # Use high risk time points as reference
        time_points = km_data['high_risk']['times']
    
    # Get baseline survival curves
    low_risk_times = km_data['low_risk']['times']
    low_risk_surv = km_data['low_risk']['surv']
    high_risk_times = km_data['high_risk']['times']
    high_risk_surv = km_data['high_risk']['surv']
    
    # Interpolate survival to common time points
    f_low = interp1d(low_risk_times, low_risk_surv, 
                     kind='linear', fill_value='extrapolate')
    f_high = interp1d(high_risk_times, high_risk_surv, 
                      kind='linear', fill_value='extrapolate')
    
    # Get survival at common time points
    surv_low = f_low(time_points)
    surv_high = f_high(time_points)
    
    # Normalize risk score to [0,1] for interpolation
    median_cutoff = km_data['median_cutoff']
    min_risk = km_data.get('min_risk', -1.5)  # Default if not in data
    max_risk = km_data.get('max_risk', 1.5)   # Default if not in data
    
    # Convert risk score to weight for blending
    # risk_score < median_cutoff -> closer to low risk
    # risk_score > median_cutoff -> closer to high risk
    if risk_score <= median_cutoff:
        # Between min_risk and median
        if median_cutoff - min_risk > 0:
            weight = (risk_score - min_risk) / (median_cutoff - min_risk)
        else:
            weight = 0
        weight = max(0, min(1, weight))  # Clamp to [0,1]
        # Lower weight means closer to low risk
        patient_surv = (1 - weight) * surv_low + weight * surv_high
    else:
        # Between median and max_risk
        if max_risk - median_cutoff > 0:
            weight = (risk_score - median_cutoff) / (max_risk - median_cutoff)
        else:
            weight = 0
        weight = max(0, min(1, weight))  # Clamp to [0,1]
        # Higher weight means closer to high risk
        patient_surv = (1 - weight) * surv_low + weight * surv_high
    
    # Ensure survival doesn't exceed 1 or go below 0
    patient_surv = np.clip(patient_surv, 0, 1)
    
    return time_points, patient_surv

def get_survival_at_time(time_points, surv_curve, target_time):
    """Get survival probability at specific time point"""
    # Find closest time point
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
        <p style='font-size: 1.1rem;'>Dynamic Risk Score with Personalized Survival Curves</p>
        <p style='font-size: 0.9rem;'>C-index: 0.8321 | iAUC: 0.8940</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("Test C-index", "0.8321")
        st.metric("iAUC", "0.8940")
        st.metric("Risk Group Cutoff", "-0.3187")
        
        st.markdown("---")
        st.header("📋 Patient Instructions")
        st.markdown("""
        1. Enter patient characteristics
        2. Click 'Predict Survival'
        3. View dynamic risk score
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
            # Collect features
            feature_values = []
            
            st.markdown("#### Patient Status")
            eco = st.selectbox(
                "ECOG Performance Status",
                ["ECOG 1-2 (Good)", "ECOG 3-4 (Poor)"],
                help="ECOG 3-4 indicates poor functional status"
            )
            feature_values.append(1 if "3-4" in eco else 0)
            
            st.markdown("#### Disease Characteristics")
            bone_met = st.selectbox(
                "Bone Metastases",
                ["No", "Yes"],
                help="Presence of bone metastases"
            )
            feature_values.append(1 if bone_met == "Yes" else 0)
            
            mult_met = st.selectbox(
                "Multiple Metastases",
                ["No", "Yes"],
                help="Multiple metastasis sites"
            )
            feature_values.append(1 if mult_met == "Yes" else 0)
            
            stage_3 = st.selectbox(
                "Stage III Disease",
                ["No", "Yes"]
            )
            feature_values.append(1 if stage_3 == "Yes" else 0)
            
            stage_4 = st.selectbox(
                "Stage IV Disease",
                ["No", "Yes"]
            )
            feature_values.append(1 if stage_4 == "Yes" else 0)
            
            histology = st.selectbox(
                "Histologic Type",
                ["Well differentiated", "Poorly differentiated"]
            )
            feature_values.append(1 if "Poorly" in histology else 0)
            
            st.markdown("#### Treatment")
            adt = st.selectbox(
                "ADT Therapy",
                ["No", "Yes"]
            )
            feature_values.append(1 if adt == "Yes" else 0)
            
            radio = st.selectbox(
                "Radiotherapy",
                ["No", "Yes"]
            )
            feature_values.append(1 if radio == "Yes" else 0)
            
            st.markdown("#### Biomarkers")
            psa_nadir = st.selectbox(
                "PSA Nadir",
                ["Low", "High"]
            )
            feature_values.append(1 if psa_nadir == "High" else 0)
            
            time_to_nadir = st.selectbox(
                "Time to PSA Nadir",
                ["Low", "High"]
            )
            feature_values.append(1 if time_to_nadir == "High" else 0)
            
            st.markdown("---")
            
            # Time selection for prediction
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
            
            # Make prediction
            risk_score = predict_risk_score(
                feature_values,
                models['model'],
                models['scaler']
            )
            
            # Generate patient-specific survival curve
            time_points, patient_surv = get_patient_survival_curve(
                risk_score,
                models['km_data']
            )
            
            # Get survival at specific time point
            surv_prob = get_survival_at_time(time_points, patient_surv, prediction_time)
            
            # Determine risk group
            is_high_risk = risk_score >= models['km_data']['median_cutoff']
            risk_group = "High Risk" if is_high_risk else "Low Risk"
            
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Dynamic Risk Score",
                    f"{risk_score:.3f}",
                    delta=None
                )
            
            with col_b:
                st.metric(
                    "Risk Group",
                    risk_group,
                    delta=None
                )
            
            with col_c:
                st.metric(
                    f"Survival at {prediction_time}m",
                    f"{surv_prob:.1%}",
                    delta=None
                )
            
            # Risk group indicator
            if risk_group == "High Risk":
                st.error("⚠️ HIGH RISK PATIENT - Consider aggressive treatment approach")
            else:
                st.success("✅ LOW RISK PATIENT - Standard treatment approach appropriate")
            
            # Dynamic KM Plot with 3 curves
            st.subheader("📈 Dynamic Survival Curves")
            
            # Get KM data
            km_data = models['km_data']
            
            # Create interactive plot with 3 curves
            fig = go.Figure()
            
            # Low risk curve
            fig.add_trace(go.Scatter(
                x=km_data['low_risk']['times'],
                y=km_data['low_risk']['surv'],
                mode='lines',
                name=f"Low Risk (n={km_data['n_low']})",
                line=dict(color='#2E86AB', width=3, dash='dash')
            ))
            
            # High risk curve
            fig.add_trace(go.Scatter(
                x=km_data['high_risk']['times'],
                y=km_data['high_risk']['surv'],
                mode='lines',
                name=f"High Risk (n={km_data['n_high']})",
                line=dict(color='#E63946', width=3, dash='dash')
            ))
            
            # Patient-specific curve (dynamic)
            fig.add_trace(go.Scatter(
                x=time_points,
                y=patient_surv,
                mode='lines',
                name=f'Patient (Risk: {risk_score:.3f})',
                line=dict(color='#FFD700', width=4)
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
            
            # Add shaded region to show dynamic range
            fig.add_trace(go.Scatter(
                x=list(km_data['low_risk']['times']) + list(km_data['high_risk']['times'])[::-1],
                y=list(km_data['low_risk']['surv']) + list(km_data['high_risk']['surv'])[::-1],
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Risk Range',
                showlegend=True
            ))
            
            # Layout
            fig.update_layout(
                title='<b>Personalized Survival Prediction</b><br><sup>Patient curve interpolated between risk groups</sup>',
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
            
            # Add grid
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Dynamic risk score interpretation
            st.markdown("---")
            st.subheader("📊 Risk Score Interpretation")
            
            # Create a risk score gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_score,
                title = {'text': "Dynamic Risk Score"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [-1.5, 1.5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [-1.5, -0.3187], 'color': "#2E86AB"},
                        {'range': [-0.3187, 1.5], 'color': "#E63946"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': -0.3187
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Additional dynamic information
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.info(f"""
                **Risk Score Components:**
                - Current Score: {risk_score:.3f}
                - Cutoff: {km_data['median_cutoff']:.3f}
                - Distance from cutoff: {abs(risk_score - km_data['median_cutoff']):.3f}
                - Risk Level: {risk_group}
                """)
            
            with col_info2:
                # Calculate relative risk
                if risk_score <= km_data['median_cutoff']:
                    rel_risk = 0.5 * (1 + (risk_score - km_data['median_cutoff']) / 
                                    abs(km_data['median_cutoff'] - min(km_data.get('min_risk', -1.5))))
                    rel_risk = max(0, min(1, rel_risk))
                    risk_text = f"Low Risk (Relative Risk: {rel_risk:.1%})"
                else:
                    rel_risk = 0.5 * (1 + (risk_score - km_data['median_cutoff']) / 
                                    abs(max(km_data.get('max_risk', 1.5)) - km_data['median_cutoff']))
                    rel_risk = min(1, rel_risk)
                    risk_text = f"High Risk (Relative Risk: {rel_risk:.1%})"
                
                st.success(f"""
                **Risk Assessment:**
                - {risk_text}
                - Survival advantage vs high risk: {(patient_surv - km_data['high_risk']['surv'][min(range(len(km_data['high_risk']['times'])), key=lambda i: abs(km_data['high_risk']['times'][i] - prediction_time))]):.1%}
                - Survival disadvantage vs low risk: {(km_data['low_risk']['surv'][min(range(len(km_data['low_risk']['times'])), key=lambda i: abs(km_data['low_risk']['times'][i] - prediction_time))] - patient_surv):.1%}
                """)
            
            st.markdown("""
            **Model Performance Context:**
            - This model has excellent discrimination (C-index: 0.8321)
            - The risk group separation is highly significant (p < 0.0001)
            - The dynamic curve represents personalized survival probability
            - Results should be interpreted in clinical context
            """)
            
        else:
            # Welcome message
            st.info("👈 Enter patient characteristics in the left panel and click 'Predict Survival'")
            
            # Show example
            with st.expander("🔍 View Example Patient"):
                st.markdown("""
                **Example: 65-year-old male**
                - ECOG: 1-2 (Good)
                - No bone metastases
                - No multiple metastases
                - Stage II disease
                - Well differentiated histology
                - Received ADT and radiotherapy
                - PSA Nadir: Low
                - Time to PSA Nadir: Low
                
                **Expected Results:**
                - Dynamic Risk Score: ~-0.85
                - Risk Group: Low Risk
                - 36-month survival: ~82%
                - Personalized curve shown in plot
                """)

if __name__ == "__main__":
    main()