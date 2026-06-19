# =========================================
# PROSTATE CANCER SURVIVAL PREDICTOR WEB APP
# GBSA Model with RSF Features
# Test C-index: 0.8321
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
# PREDICTION FUNCTIONS
# =========================================

def predict_risk_score(features, model, scaler):
    """Predict risk score from patient features"""
    feature_array = np.array(features).reshape(1, -1)
    feature_scaled = scaler.transform(feature_array)
    risk_score = model.predict(feature_scaled)[0]
    return risk_score

def get_survival_probability(risk_score, time, km_data):
    """Get survival probability at time T for risk group"""
    is_high_risk = risk_score >= km_data['median_cutoff']
    
    if is_high_risk:
        times = km_data['high_risk']['times']
        surv = km_data['high_risk']['surv']
        group = 'High Risk'
    else:
        times = km_data['low_risk']['times']
        surv = km_data['low_risk']['surv']
        group = 'Low Risk'
    
    # Find closest time point
    idx = min(range(len(times)), key=lambda i: abs(times[i] - time))
    
    return surv[idx], group

# =========================================
# MAIN APPLICATION
# =========================================

def main():
    # Title
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background-color: #1B4332; border-radius: 10px; color: white;'>
        <h1>🩺 Prostate Cancer Survival Predictor</h1>
        <p style='font-size: 1.1rem;'>Powered by GBSA Model with RSF Features</p>
        <p style='font-size: 0.9rem;'>C-index: 0.8321 | iAUC: 0.8940</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Performance")
        st.metric("Test C-index", "0.8321", "Excellent")
        st.metric("iAUC", "0.8940", "Excellent")
        st.metric("Risk Group Cutoff", "-0.3187")
        
        st.markdown("---")
        st.header("📋 Patient Instructions")
        st.markdown("""
        1. Enter patient characteristics
        2. Select prediction time
        3. Click 'Predict Survival'
        4. View results and visualization
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
            
            # Get survival probability
            surv_prob, risk_group = get_survival_probability(
                risk_score,
                prediction_time,
                models['km_data']
            )
            
            # Display metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Risk Score",
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
            
            # KM Plot
            st.subheader("📈 Patient Position on KM Curves")
            
            # Get KM data
            km_data = models['km_data']
            
            # Create interactive plot
            fig = go.Figure()
            
            # Low risk curve
            fig.add_trace(go.Scatter(
                x=km_data['low_risk']['times'],
                y=km_data['low_risk']['surv'],
                mode='lines',
                name=f"Low Risk (n={km_data['n_low']})",
                line=dict(color='#2E86AB', width=3)
            ))
            
            # High risk curve
            fig.add_trace(go.Scatter(
                x=km_data['high_risk']['times'],
                y=km_data['high_risk']['surv'],
                mode='lines',
                name=f"High Risk (n={km_data['n_high']})",
                line=dict(color='#E63946', width=3)
            ))
            
            # Patient point
            fig.add_trace(go.Scatter(
                x=[prediction_time],
                y=[surv_prob],
                mode='markers',
                name='Patient',
                marker=dict(
                    size=25,
                    color='#FFD700',
                    symbol='star',
                    line=dict(width=3, color='black')
                )
            ))
            
            # Layout
            fig.update_layout(
                title='Kaplan-Meier Survival Curves',
                xaxis_title='Time (months)',
                yaxis_title='Survival Probability',
                yaxis_range=[0, 1.02],
                xaxis_range=[0, 62],
                hovermode='x',
                legend=dict(
                    x=0.02,
                    y=0.98,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='black',
                    borderwidth=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Clinical interpretation
            with st.expander("📖 Clinical Interpretation", expanded=True):
                st.markdown(f"""
                **Patient Summary:**
                - Risk Score: {risk_score:.3f}
                - Risk Group: **{risk_group}**
                - Predicted Survival at {prediction_time} months: **{surv_prob:.1%}**
                
                **Clinical Recommendations:**
                """)
                
                if risk_group == "High Risk":
                    st.markdown("""
                    - ⚠️ **Intensify treatment**: Consider combination therapies
                    - 📊 **Close monitoring**: More frequent follow-up visits
                    - 🏥 **Clinical trials**: Evaluate eligibility for experimental therapies
                    - 📈 **Median survival**: ~38.0 months for this group
                    """)
                else:
                    st.markdown("""
                    - ✅ **Standard treatment**: Current treatment plan appropriate
                    - 📊 **Regular monitoring**: Standard follow-up schedule
                    - 🏥 **Quality of life**: Focus on maintaining quality of life
                    - 📈 **Median survival**: ~54.0 months for this group
                    """)
                
                st.markdown(f"""
                **Model Performance Context:**
                - This model has excellent discrimination (C-index: 0.8321)
                - The risk group separation is highly significant (p < 0.0001)
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
                - Risk Score: ~-0.85
                - Risk Group: Low Risk
                - 36-month survival: ~82%
                """)

if __name__ == "__main__":
    main()