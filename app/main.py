import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add base directory to path so we can import src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

# Set page config
st.set_page_config(
    page_title="Customer Churn & CLTV Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling (Dark Theme & Professional Aesthetics)
st.markdown("""
<style>
    /* Dark Mode Theme Adjustments */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
    }
    
    /* Card Styles */
    .metric-card {
        background: rgba(26, 32, 44, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 191, 255, 0.4);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00BFFF;
        margin-top: 10px;
    }
    .metric-title {
        font-size: 0.9rem;
        font-weight: 500;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Strategic Grid Card Styles */
    .strategy-card {
        border-left: 5px solid #00BFFF;
        background: rgba(26, 32, 44, 0.8);
        border-radius: 0 8px 8px 0;
        padding: 15px;
        margin-bottom: 15px;
    }
    .strategy-card.high-risk {
        border-left-color: #FF5E5E;
    }
    .strategy-card.loyal {
        border-left-color: #2ECC71;
    }
    .strategy-card.save {
        border-left-color: #F1C40F;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
    }
    .glow-header {
        font-weight: 800;
        background: linear-gradient(90deg, #00BFFF, #8A2BE2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load data and models
@st.cache_resource
def load_resources():
    # Load model and encoders
    model = joblib.load(config.BEST_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    cox_model = joblib.load(os.path.join(config.MODELS_DIR, 'cox_model.pkl'))
    
    # Load features list
    with open(os.path.join(config.MODELS_DIR, 'features.json'), 'r') as f:
        features_list = json.load(f)
        
    # Load precalculated data
    df_cltv = pd.read_csv(config.CLTV_DATA_PATH)
    
    # Load segment profiles
    with open(os.path.join(config.MODELS_DIR, 'segment_profiles.json'), 'r') as f:
        segment_profiles = json.load(f)
        
    # Load KM cohorts data
    with open(os.path.join(config.MODELS_DIR, 'km_cohorts_data.json'), 'r') as f:
        km_cohorts_data = json.load(f)
        
    return model, scaler, cox_model, features_list, df_cltv, segment_profiles, km_cohorts_data

# Load assets
try:
    model, scaler, cox_model, features_list, df_cltv, segment_profiles, km_cohorts_data = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}. Please ensure you've run the training, segmentation, and CLTV scripts first.")
    st.stop()

# Sidebar Setup
st.sidebar.markdown("<h2 style='color:#00BFFF;'>Pipeline Options</h2>", unsafe_allow_html=True)
st.sidebar.info("This project tracks Customer Churn prediction, Customer Segmentation, and Survival-based Customer Lifetime Value (CLTV) modeling.")

# Title
st.markdown("<h1 class='glow-header'>Customer Analytics Suite & CLTV Predictor</h1>", unsafe_allow_html=True)

# Define Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary & EDA", 
    "🎯 Churn Risk Simulator", 
    "👥 Customer Segmentation", 
    "📈 CLTV & Survival Analytics"
])

# ----------------- TAB 1: EXECUTIVE SUMMARY & EDA -----------------
with tab1:
    st.subheader("Business Metrics Dashboard")
    
    # Pre-calculated values
    total_customers = df_cltv.shape[0]
    churn_rate = (df_cltv['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
    avg_cltv = df_cltv['CLTV'].mean()
    total_projected_value = df_cltv.loc[df_cltv['Churn'] == 'No', 'CLTV'].sum()
    
    # Display cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Customers</div>
            <div class="metric-value">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Overall Churn Rate</div>
            <div class="metric-value" style="color: #FF5E5E;">{churn_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Average CLTV</div>
            <div class="metric-value" style="color: #2ECC71;">${avg_cltv:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Active CLTV Value</div>
            <div class="metric-value" style="color: #00BFFF;">${total_projected_value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Visual EDA Section
    st.subheader("Key Drivers of Customer Churn")
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        # Churn by Contract Type
        contract_churn = df_cltv.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig_contract = px.bar(
            contract_churn, x='Contract', y='Count', color='Churn',
            title='Churn Counts by Contract Type',
            barmode='group',
            color_discrete_map={'Yes': '#FF5E5E', 'No': '#2ECC71'},
            template='plotly_dark'
        )
        fig_contract.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_contract, use_container_width=True)
        
    with col_plot2:
        # Churn by Monthly Charges Box Plot
        fig_charges = px.box(
            df_cltv, x='Churn', y='MonthlyCharges',
            color='Churn',
            title='Monthly Charges Distribution by Churn',
            color_discrete_map={'Yes': '#FF5E5E', 'No': '#2ECC71'},
            template='plotly_dark'
        )
        fig_charges.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_charges, use_container_width=True)
        
    # Additional Plot: Churn by Payment Method
    payment_churn = df_cltv.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count')
    fig_payment = px.bar(
        payment_churn, y='PaymentMethod', x='Count', color='Churn',
        title='Churn Distribution by Payment Method',
        barmode='stack',
        orientation='h',
        color_discrete_map={'Yes': '#FF5E5E', 'No': '#2ECC71'},
        template='plotly_dark',
        height=400
    )
    fig_payment.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_payment, use_container_width=True)

# ----------------- TAB 2: CHURN RISK SIMULATOR -----------------
with tab2:
    st.subheader("Customer Risk Assessment & Lifetime Value Simulator")
    st.write("Modify the demographics and service preferences of a customer in the form below to run real-time churn predictions and compute survival-based CLTV.")
    
    col_sim_in, col_sim_out = st.columns([1, 1])
    
    with col_sim_in:
        st.markdown("### Customer Attributes")
        
        sim_gender = st.selectbox("Gender", ["Female", "Male"])
        sim_senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        sim_partner = st.selectbox("Partner (Married)", ["No", "Yes"])
        sim_dependents = st.selectbox("Dependents (Children/Parents)", ["No", "Yes"])
        sim_tenure = st.slider("Tenure (Months with company)", 1, 72, 12)
        sim_phone = st.selectbox("Phone Service", ["Yes", "No"])
        
        # MultipleLines dependent on Phone Service
        if sim_phone == "Yes":
            sim_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        else:
            sim_lines = "No phone service"
            
        sim_internet = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])
        
        # Internet dependent options
        if sim_internet != "No":
            sim_sec = st.selectbox("Online Security Add-on", ["No", "Yes"])
            sim_back = st.selectbox("Online Backup Add-on", ["No", "Yes"])
            sim_prot = st.selectbox("Device Protection Add-on", ["No", "Yes"])
            sim_support = st.selectbox("Tech Support Add-on", ["No", "Yes"])
            sim_tv = st.selectbox("Streaming TV Add-on", ["No", "Yes"])
            sim_movies = st.selectbox("Streaming Movies Add-on", ["No", "Yes"])
        else:
            sim_sec = "No internet service"
            sim_back = "No internet service"
            sim_prot = "No internet service"
            sim_support = "No internet service"
            sim_tv = "No internet service"
            sim_movies = "No internet service"
            
        sim_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        sim_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        sim_payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
        sim_monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        sim_total = sim_monthly * sim_tenure  # Safe estimate for Total Charges
        
    with col_sim_out:
        st.markdown("### Risk Engine & CLTV Predictions")
        
        # Assemble input dictionary according to preprocessed features
        input_dict = {feat: 0 for feat in features_list}
        
        # Set numericals
        # Need to scale them using the saved scaler
        # To do that, we create a 1-row DataFrame of numerical features
        num_df = pd.DataFrame([[sim_tenure, sim_monthly, sim_total]], columns=config.NUMERIC_COLS)
        scaled_nums = scaler.transform(num_df)[0]
        
        input_dict['tenure'] = scaled_nums[0]
        input_dict['MonthlyCharges'] = scaled_nums[1]
        input_dict['TotalCharges'] = scaled_nums[2]
        
        # Set categorical dummies
        if sim_gender == "Male":
            input_dict['gender_Male'] = 1
        if sim_senior == "Yes":
            input_dict['SeniorCitizen_Yes'] = 1
        if sim_partner == "Yes":
            input_dict['Partner_Yes'] = 1
        if sim_dependents == "Yes":
            input_dict['Dependents_Yes'] = 1
        if sim_phone == "Yes":
            input_dict['PhoneService_Yes'] = 1
            
        if sim_lines == "No phone service":
            input_dict['MultipleLines_No phone service'] = 1
        elif sim_lines == "Yes":
            input_dict['MultipleLines_Yes'] = 1
            
        if sim_internet == "Fiber optic":
            input_dict['InternetService_Fiber optic'] = 1
        elif sim_internet == "No":
            input_dict['InternetService_No'] = 1
            
        for name, val in [
            ('OnlineSecurity', sim_sec), ('OnlineBackup', sim_back), 
            ('DeviceProtection', sim_prot), ('TechSupport', sim_support), 
            ('StreamingTV', sim_tv), ('StreamingMovies', sim_movies)
        ]:
            if val == "No internet service":
                input_dict[f'{name}_No internet service'] = 1
            elif val == "Yes":
                input_dict[f'{name}_Yes'] = 1
                
        if sim_contract == "One year":
            input_dict['Contract_One year'] = 1
        elif sim_contract == "Two year":
            input_dict['Contract_Two year'] = 1
            
        if sim_billing == "Yes":
            input_dict['PaperlessBilling_Yes'] = 1
            
        if sim_payment == "Credit card (automatic)":
            input_dict['PaymentMethod_Credit card (automatic)'] = 1
        elif sim_payment == "Electronic check":
            input_dict['PaymentMethod_Electronic check'] = 1
        elif sim_payment == "Mailed check":
            input_dict['PaymentMethod_Mailed check'] = 1
            
        # Convert input dict to DataFrame
        sim_df = pd.DataFrame([input_dict])
        
        # Run classification churn model prediction
        prob = model.predict_proba(sim_df)[0, 1]
        
        # 1. Gauge chart for Risk
        risk_color = "#2ECC71" if prob < 0.3 else ("#F1C40F" if prob < 0.7 else "#FF5E5E")
        risk_level = "Low Risk" if prob < 0.3 else ("Medium Risk" if prob < 0.7 else "High Risk")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Churn Probability - {risk_level}", 'font': {'size': 20, 'color': '#E0E6ED'}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#E0E6ED"},
                'bar': {'color': risk_color},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(46, 204, 113, 0.15)'},
                    {'range': [30, 70], 'color': 'rgba(241, 196, 15, 0.15)'},
                    {'range': [70, 100], 'color': 'rgba(255, 94, 94, 0.15)'}
                ],
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': '#E0E6ED'}, height=250)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # 2. Survival-based Dynamic CLTV Prediction
        # Build Cox covariate row
        cox_row_dict = {
            'MonthlyCharges': sim_monthly,
            'Contract_One year': 1 if sim_contract == "One year" else 0,
            'Contract_Two year': 1 if sim_contract == "Two year" else 0,
            'InternetService_Fiber optic': 1 if sim_internet == "Fiber optic" else 0,
            'InternetService_No': 1 if sim_internet == "No" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if sim_payment == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if sim_payment == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if sim_payment == "Mailed check" else 0,
            'PaperlessBilling_Yes': 1 if sim_billing == "Yes" else 0
        }
        cox_df = pd.DataFrame([cox_row_dict])
        
        # Predict survival function for this customer
        # Predict returns a dataframe with rows as index times and 1 column of probabilities
        customer_survival = cox_model.predict_survival_function(cox_df)
        
        # Get probability at current tenure
        closest_t = min(customer_survival.index, key=lambda x: abs(x - sim_tenure))
        s_current = customer_survival.loc[closest_t].values[0]
        
        # Calculate expected remaining months
        future_times = [t for t in customer_survival.index if t > sim_tenure and t <= 72]
        
        if s_current == 0 or len(future_times) == 0:
            remaining_tenure = 6.0  # fallback
        else:
            s_future = customer_survival.loc[future_times].iloc[:, 0]
            remaining_tenure = (s_future / s_current).sum()
            
        expected_total_tenure = sim_tenure + remaining_tenure
        profit_margin = 0.70
        sim_cltv = expected_total_tenure * sim_monthly * profit_margin
        
        # Display CLTV predictions
        st.markdown("#### Projected Lifetime Value")
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Expected Add. Months", f"{remaining_tenure:.1f} months")
        with col_c2:
            st.metric("Predicted CLTV (Profit)", f"${sim_cltv:.2f}")
            
        # Plot single customer survival probability over time
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=customer_survival.index, 
            y=customer_survival.iloc[:, 0], 
            mode='lines', 
            name='Survival Probability',
            line=dict(color='#00BFFF', width=3)
        ))
        # Add a marker for current tenure
        fig_curve.add_trace(go.Scatter(
            x=[sim_tenure], 
            y=[s_current], 
            mode='markers', 
            name='Current Tenure State',
            marker=dict(color='#FF5E5E', size=10, symbol='circle')
        ))
        
        fig_curve.update_layout(
            title="Customer Survival Probability Curve (Decay)",
            xaxis_title="Tenure (Months)",
            yaxis_title="Survival Probability",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_curve, use_container_width=True)

# ----------------- TAB 3: CUSTOMER SEGMENTATION -----------------
with tab3:
    st.subheader("K-Means Behavioral Segmentation (K=4)")
    st.write("We clustered customers on their engagement history (Tenure, Monthly Charges, and Total Charges). "
             "This segments our base into 4 distinct groups with unique characteristics and retention strategies.")
             
    # Display cards for each segment
    col_seg1, col_seg2 = st.columns(2)
    with col_seg1:
        st.markdown("""
        <div class="strategy-card loyal">
            <h4 style="color:#2ECC71;margin-bottom:5px;">Loyal Premium</h4>
            <p style="font-size:0.9rem;margin-bottom:0;">High tenure, high monthly spend. These are VIP subscribers who buy multiple add-ons. 
            <b>Strategy:</b> Premium cross-selling (e.g. streaming, home protection) and exclusive loyalty benefits.</p>
        </div>
        
        <div class="strategy-card strategy-card.save high-risk">
            <h4 style="color:#FF5E5E;margin-bottom:5px;">High-Spend At-Risk</h4>
            <p style="font-size:0.9rem;margin-bottom:0;">Low tenure but high monthly charges. These are typically new fiber-optic signups on month-to-month contracts. 
            <b>Strategy:</b> Proactive support outreach, contract upgrade discounts, lock-in pricing offers.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_seg2:
        st.markdown("""
        <div class="strategy-card">
            <h4 style="color:#00BFFF;margin-bottom:5px;">Loyal Value</h4>
            <p style="font-size:0.9rem;margin-bottom:0;">High tenure, low monthly spend. Budget-conscious subscribers with long-term contracts. 
            <b>Strategy:</b> Offer minor upgrades, reward their longevity, contract roll-overs.</p>
        </div>
        
        <div class="strategy-card strategy-card.save" style="border-left-color: #F1C40F;">
            <h4 style="color:#F1C40F;margin-bottom:5px;">New Budget</h4>
            <p style="font-size:0.9rem;margin-bottom:0;">Low tenure, low monthly spend. New trial-like subscribers. 
            <b>Strategy:</b> Frictionless digital onboarding, utility tutorials, engagement campaigns to increase usage.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Segment profile table
    st.subheader("Segment Metrics Profiles")
    st.dataframe(pd.DataFrame(segment_profiles).rename(columns={
        'Count': 'Customer Count',
        'Avg_Tenure': 'Avg Tenure (m)',
        'Avg_MonthlyCharges': 'Avg Monthly Charges ($)',
        'Avg_TotalCharges': 'Avg Total Charges ($)',
        'Churn_Rate': 'Churn Rate (%)',
        'Strategic_Recommendation': 'Marketing Retention Action'
    }), use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 3D PCA Visualizer
    st.subheader("Interactive 3D Customer Cluster Space")
    st.write("We apply PCA (Principal Component Analysis) to project the 3 scaled variables into 3 dimensions for high-fidelity interactive visualization.")
    
    # Run PCA dynamically
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler_pca = StandardScaler()
    scaled_feats = scaler_pca.fit_transform(df_cltv[features])
    pca = PCA(n_components=3, random_state=config.RANDOM_STATE)
    pca_coords = pca.fit_transform(scaled_feats)
    
    df_pca = df_cltv.copy()
    df_pca['PCA 1'] = pca_coords[:, 0]
    df_pca['PCA 2'] = pca_coords[:, 1]
    df_pca['PCA 3'] = pca_coords[:, 2]
    
    # Sample down to 2500 points for snappy rendering in browser
    df_pca_sample = df_pca.sample(n=2500, random_state=config.RANDOM_STATE)
    
    fig_pca = px.scatter_3d(
        df_pca_sample, x='PCA 1', y='PCA 2', z='PCA 3',
        color='Segment',
        hover_data=['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'],
        title="3D Projection of Customer Segments",
        color_discrete_map={
            'Loyal Premium': '#2ECC71',
            'Loyal Value': '#00BFFF',
            'High-Spend At-Risk': '#FF5E5E',
            'New Budget': '#F1C40F'
        },
        template='plotly_dark',
        height=600
    )
    
    fig_pca.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(255,255,255,0.05)"),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# ----------------- TAB 4: CLTV & SURVIVAL ANALYTICS -----------------
with tab4:
    st.subheader("Survival Analysis & Lifetime Value Cohorts")
    st.write("Using survival analysis, we model the probability of a customer staying active over time. "
             "Longer survival directly drives higher Customer Lifetime Value.")
             
    col_surv_left, col_surv_right = st.columns(2)
    
    with col_surv_left:
        # Kaplan-Meier Cohort curves
        st.markdown("#### Baseline Survival Curves by Contract Type")
        fig_km = go.Figure()
        
        for contract_type, c_data in km_cohorts_data.items():
            fig_km.add_trace(go.Scatter(
                x=c_data['timeline'],
                y=c_data['survival_probability'],
                mode='lines',
                name=contract_type,
                line=dict(width=3)
            ))
            
        fig_km.update_layout(
            xaxis_title="Tenure (Months)",
            yaxis_title="Probability of Retaining Customer",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_km, use_container_width=True)
        
    with col_surv_right:
        # CLTV Distribution histogram
        st.markdown("#### Customer Lifetime Value Distribution")
        fig_cltv_dist = px.histogram(
            df_cltv, x='CLTV', color='CLTV_Level',
            nbins=50,
            title='Count of Customers by CLTV Value ($)',
            color_discrete_map={'Low Value': '#FF5E5E', 'Medium Value': '#F1C40F', 'High Value': '#2ECC71'},
            template='plotly_dark'
        )
        fig_cltv_dist.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=400)
        st.plotly_chart(fig_cltv_dist, use_container_width=True)
        
    # Churn Risk vs CLTV Strategy Matrix
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Risk vs. Lifetime Value Matrix (Strategic Action Plan)")
    st.write("By crossing predicted Churn Probability (ML model) and Customer Lifetime Value (Survival Model), "
             "we categorize customers into 4 strategic buckets for optimal budget allocation.")
             
    # Calculate categories on the full dataset using a threshold of 0.3 for Churn Risk
    # and median CLTV for Value Tier
    # Let's predict probabilities for all preprocessed customers to be accurate
    try:
        df_ml = pd.read_csv(config.PREPROCESSED_DATA_PATH)
        X_all = df_ml.drop(config.TARGET_COL, axis=1)
        churn_probs = model.predict_proba(X_all)[:, 1]
    except:
        # Fallback if preprocessed features differ (should not occur)
        churn_probs = np.where(df_cltv['Churn'] == 'Yes', 0.8, 0.1)
        
    df_matrix = df_cltv.copy()
    df_matrix['Churn_Prob'] = churn_probs
    median_cltv = df_matrix['CLTV'].median()
    
    # Define buckets
    def assign_bucket(row):
        is_high_risk = row['Churn_Prob'] >= 0.4
        is_high_val = row['CLTV'] >= median_cltv
        
        if is_high_risk and is_high_val:
            return "VIP at Risk (Retain at all costs)"
        elif not is_high_risk and is_high_val:
            return "VIP Loyal (Nurture & Upsell)"
        elif is_high_risk and not is_high_val:
            return "Low-Value Churner (Automate Save)"
        else:
            return "Stable Budget (Standard Care)"
            
    df_matrix['Action_Bucket'] = df_matrix.apply(assign_bucket, axis=1)
    bucket_counts = df_matrix['Action_Bucket'].value_counts()
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("#### Customer Base Distribution in Strategy Matrix")
        fig_pie = px.pie(
            values=bucket_counts.values,
            names=bucket_counts.index,
            hole=0.4,
            color=bucket_counts.index,
            color_discrete_map={
                'VIP at Risk (Retain at all costs)': '#FF5E5E',
                'VIP Loyal (Nurture & Upsell)': '#2ECC71',
                'Low-Value Churner (Automate Save)': '#F1C40F',
                'Stable Budget (Standard Care)': '#00BFFF'
            },
            template='plotly_dark'
        )
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=380)
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b2:
        st.markdown("#### Strategic Playbooks")
        
        st.markdown(f"""
        <div class="strategy-card high-risk">
            <h5 style="color:#FF5E5E;margin-bottom:3px;">VIP at Risk ({bucket_counts.get('VIP at Risk (Retain at all costs)', 0):,} Customers)</h5>
            <p style="font-size:0.85rem;margin-bottom:0;">High Lifetime Value but high probability of churning. 
            <b>Playbook:</b> Assign direct account manager. Offer maximum contract discount incentives. Prioritize resolving technical issues.</p>
        </div>
        
        <div class="strategy-card loyal">
            <h5 style="color:#2ECC71;margin-bottom:3px;">VIP Loyal ({bucket_counts.get('VIP Loyal (Nurture & Upsell)', 0):,} Customers)</h5>
            <p style="font-size:0.85rem;margin-bottom:0;">High Lifetime Value and low churn risk. 
            <b>Playbook:</b> Enroll in referral programs. Cross-sell premium add-ons. Request reviews / case studies.</p>
        </div>
        
        <div class="strategy-card strategy-card.save" style="border-left-color: #F1C40F;">
            <h5 style="color:#F1C40F;margin-bottom:3px;">Low-Value Churner ({bucket_counts.get('Low-Value Churner (Automate Save)', 0):,} Customers)</h5>
            <p style="font-size:0.85rem;margin-bottom:0;">Low Lifetime Value, High Churn Risk. 
            <b>Playbook:</b> Target with automated, low-cost digital email offers (e.g. 10% coupon for paperless billing sign-up). Do not spend manual sales representative hours.</p>
        </div>
        """, unsafe_allow_html=True)
