import os
import sys
import pandas as pd
import numpy as np
import joblib
import json
import streamlit as st
import shap
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

# ==================== PREMIUM DESIGN SYSTEM ====================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    /* ===== ANIMATIONS ===== */
    @keyframes pulse-risk {
        0% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(248, 113, 113, 0); }
        100% { box-shadow: 0 0 0 0 rgba(248, 113, 113, 0); }
    }
    
    @keyframes fade-in-up {
        0% { opacity: 0; transform: translateY(15px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes gradient-sweep {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ===== GLOBAL RESET & TYPOGRAPHY ===== */
    *, *::before, *::after { box-sizing: border-box; }
    
    .stApp {
        background: linear-gradient(160deg, #0a0e17 0%, #0d1321 40%, #111827 100%);
        color: #c9d1d9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Apply fade-in animation to main content blocks */
    div[data-testid="stVerticalBlock"] > div {
        animation: fade-in-up 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #e6edf3 !important;
        letter-spacing: -0.02em;
    }
    
    label, p {
        font-family: 'Inter', sans-serif;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.4); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.7); }
    
    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #131b2e 100%) !important;
        border-right: 1px solid rgba(99, 102, 241, 0.15) !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #8b949e !important;
    }
    
    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(13, 19, 33, 0.6);
        border-radius: 12px;
        padding: 6px;
        border: 1px solid rgba(99, 102, 241, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 0.85rem;
        color: #8b949e;
        background: transparent;
        border: none;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #c9d1d9;
        background: rgba(99, 102, 241, 0.08);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2)) !important;
        color: #a5b4fc !important;
        font-weight: 600;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }
    
    /* ===== GLASSMORPHISM METRIC CARDS ===== */
    .glass-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.5));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(99, 102, 241, 0.12);
        border-radius: 16px;
        padding: 24px 20px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-color, #6366f1), transparent);
        border-radius: 16px 16px 0 0;
    }
    .glass-card:hover {
        transform: translateY(-6px);
        border-color: rgba(99, 102, 241, 0.35);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3), 0 0 30px rgba(99, 102, 241, 0.08);
    }
    .card-icon {
        font-size: 1.8rem;
        margin-bottom: 8px;
        display: block;
        transition: transform 0.3s ease;
    }
    .glass-card:hover .card-icon {
        transform: scale(1.15);
    }
    .card-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 6px;
    }
    .card-value {
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    /* ===== SECTION HEADERS ===== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 28px 0 18px 0;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(99, 102, 241, 0.12);
    }
    .section-header .icon {
        font-size: 1.4rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.15));
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .section-header:hover .icon {
        transform: rotate(-5deg) scale(1.05);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.3);
    }
    .section-header .text {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e6edf3;
        letter-spacing: -0.01em;
    }
    .section-desc {
        color: #8b949e;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-bottom: 20px;
    }
    
    /* ===== STRATEGY CARDS V2 ===== */
    .strat-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7), rgba(30, 41, 59, 0.4));
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-left: 4px solid var(--card-accent, #6366f1);
        border-radius: 0 12px 12px 0;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: all 0.3s ease;
    }
    .strat-card:hover {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.6));
        border-left-width: 6px;
        transform: translateX(6px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .strat-card h4 {
        font-size: 1rem;
        font-weight: 700;
        margin: 0 0 6px 0;
    }
    .strat-card p {
        font-size: 0.82rem;
        color: #8b949e;
        line-height: 1.55;
        margin: 0;
    }
    .strat-card b { color: #a5b4fc; }
    
    /* ===== NBA RESULT CARDS ===== */
    .nba-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.5));
        backdrop-filter: blur(16px);
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 14px;
        padding: 20px 22px;
        margin-top: 12px;
        transition: all 0.3s ease;
    }
    .nba-card.high-risk {
        animation: pulse-risk 2s infinite;
    }
    .nba-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    .nba-card .nba-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .nba-card .nba-text {
        font-size: 0.92rem;
        line-height: 1.6;
        color: #c9d1d9;
    }
    
    /* ===== MAIN HEADER ===== */
    .hero-header {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 25%, #c084fc 50%, #e879f9 75%, #818cf8 100%);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 6px;
        line-height: 1.2;
        animation: gradient-sweep 4s linear infinite;
    }
    .hero-subtitle {
        font-size: 0.88rem;
        color: #64748b;
        font-weight: 400;
        margin-bottom: 28px;
        letter-spacing: 0.02em;
    }
    
    /* ===== SIDEBAR BRANDING ===== */
    .sidebar-brand {
        text-align: center;
        padding: 20px 10px 24px 10px;
        border-bottom: 1px solid rgba(99, 102, 241, 0.1);
        margin-bottom: 20px;
    }
    .sidebar-brand .logo { 
        font-size: 2.4rem; 
        margin-bottom: 6px; 
        display: inline-block; 
        transition: transform 0.4s ease;
    }
    .sidebar-brand:hover .logo {
        transform: rotate(15deg) scale(1.1);
    }
    .sidebar-brand .name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e6edf3;
        letter-spacing: -0.01em;
    }
    .sidebar-brand .version {
        font-size: 0.68rem;
        color: #64748b;
        font-weight: 500;
        margin-top: 4px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .sidebar-nav {
        padding: 8px 0;
    }
    .sidebar-nav .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 12px 16px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        color: #8b949e;
        margin-bottom: 6px;
        transition: all 0.2s ease;
        cursor: default;
    }
    .sidebar-nav .nav-item:hover {
        background: rgba(99, 102, 241, 0.12);
        color: #c9d1d9;
        transform: translateX(4px);
    }
    .sidebar-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    
    /* ===== FOOTER ===== */
    .app-footer {
        text-align: center;
        padding: 24px 0;
        margin-top: 40px;
        border-top: 1px solid rgba(99, 102, 241, 0.08);
        color: #475569;
        font-size: 0.75rem;
        letter-spacing: 0.03em;
        transition: color 0.3s ease;
    }
    .app-footer:hover { color: #64748b; }
    .app-footer a { color: #818cf8; text-decoration: none; transition: color 0.3s ease; }
    .app-footer a:hover { color: #a78bfa; text-decoration: underline; }
    
    /* ===== HIDE STREAMLIT DEFAULTS ===== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    
    /* ===== FORM WIDGETS ===== */
    .stSelectbox label, .stSlider label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
    }
    div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.6) !important;
        border-color: rgba(99, 102, 241, 0.15) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(99, 102, 241, 0.4) !important;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.1);
    }
    /* Slider improvements */
    .stSlider > div > div > div > div {
        background-color: #818cf8 !important;
    }
    
    /* ===== DATAFRAME ===== */
    .stDataFrame {
        border: 1px solid rgba(99, 102, 241, 0.1) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    /* ===== RESPONSIVE FLUIDITY ===== */
    @media (max-width: 768px) {
        .hero-header { font-size: 1.8rem; }
        .hero-subtitle { font-size: 0.8rem; }
        .glass-card { padding: 16px 14px; }
        .card-value { font-size: 1.5rem; }
        .section-header .text { font-size: 1rem; }
        .section-header .icon { font-size: 1.2rem; }
        .strat-card { padding: 14px 16px; margin-bottom: 10px; }
        .nba-card { padding: 16px 18px; }
        .stTabs [data-baseweb="tab"] { padding: 8px 12px; font-size: 0.75rem; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA & MODELS ====================
@st.cache_resource
def load_resources():
    model = joblib.load(config.BEST_MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    cox_model = joblib.load(os.path.join(config.MODELS_DIR, 'cox_model.pkl'))
    
    with open(os.path.join(config.MODELS_DIR, 'features.json'), 'r') as f:
        features_list = json.load(f)
        
    df_cltv = pd.read_csv(config.CLTV_DATA_PATH)
    
    with open(os.path.join(config.MODELS_DIR, 'segment_profiles.json'), 'r') as f:
        segment_profiles = json.load(f)
        
    with open(os.path.join(config.MODELS_DIR, 'km_cohorts_data.json'), 'r') as f:
        km_cohorts_data = json.load(f)
        
    shap_explainer = joblib.load(os.path.join(config.MODELS_DIR, 'shap_explainer.pkl'))
        
    return model, scaler, cox_model, features_list, df_cltv, segment_profiles, km_cohorts_data, shap_explainer

try:
    model, scaler, cox_model, features_list, df_cltv, segment_profiles, km_cohorts_data, shap_explainer = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}. Please ensure you've run the training, segmentation, and CLTV scripts first.")
    st.stop()

# ==================== PLOTLY CHART THEME ====================
CHART_LAYOUT = dict(
    template='plotly_dark',
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#8b949e', size=12),
    title_font=dict(size=14, color='#c9d1d9', family='Inter, sans-serif'),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(gridcolor='rgba(99, 102, 241, 0.06)', zerolinecolor='rgba(99, 102, 241, 0.1)'),
    yaxis=dict(gridcolor='rgba(99, 102, 241, 0.06)', zerolinecolor='rgba(99, 102, 241, 0.1)'),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    colorway=['#818cf8', '#a78bfa', '#c084fc', '#f472b6', '#fb923c', '#34d399', '#38bdf8']
)

COLOR_MAP_CHURN = {'Yes': '#f87171', 'No': '#34d399'}
COLOR_MAP_SEGMENTS = {
    'Loyal Premium': '#34d399',
    'Loyal Value': '#38bdf8',
    'High-Spend At-Risk': '#f87171',
    'New Budget': '#fbbf24'
}

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="logo">🔬</span>
        <div class="name">ChurnGuard AI</div>
        <div class="version">Enterprise Analytics Suite v2.0</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-nav">
        <div class="nav-item">📊 Executive Dashboard</div>
        <div class="nav-item">🎯 Risk Simulator</div>
        <div class="nav-item">👥 Customer Segments</div>
        <div class="nav-item">📈 CLTV & Survival</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="padding: 12px 14px;">
        <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">Pipeline Status</div>
        <div style="font-size:0.8rem;color:#8b949e;margin-bottom:6px;">
            <span style="color:#34d399;">●</span> XGBoost Model <span class="sidebar-badge" style="background:rgba(52,211,153,0.12);color:#34d399;">Trained</span>
        </div>
        <div style="font-size:0.8rem;color:#8b949e;margin-bottom:6px;">
            <span style="color:#34d399;">●</span> SHAP Explainer <span class="sidebar-badge" style="background:rgba(52,211,153,0.12);color:#34d399;">Active</span>
        </div>
        <div style="font-size:0.8rem;color:#8b949e;margin-bottom:6px;">
            <span style="color:#38bdf8;">●</span> K-Means (K=4) <span class="sidebar-badge" style="background:rgba(56,189,248,0.12);color:#38bdf8;">Fitted</span>
        </div>
        <div style="font-size:0.8rem;color:#8b949e;">
            <span style="color:#a78bfa;">●</span> Cox PH Model <span class="sidebar-badge" style="background:rgba(167,139,250,0.12);color:#a78bfa;">Ready</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 14px;">
        <div style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">Tech Stack</div>
        <div style="font-size:0.78rem;color:#64748b;line-height:1.8;">
            XGBoost · Scikit-learn · SHAP<br>
            Lifelines · Plotly · Streamlit<br>
            K-Means · PCA · Cox PH
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== HEADER ====================
st.markdown('<div class="hero-header">ChurnGuard Analytics Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">ML-powered churn prediction · Customer segmentation · Survival-based CLTV modeling · Explainable AI</div>', unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Summary", 
    "🎯 Churn Risk Simulator", 
    "👥 Customer Segmentation", 
    "📈 CLTV & Survival Analytics",
    "🗂️ Batch CSV Scoring",
    "🛒 RFM Transaction Analytics"
])

# ==================== TAB 1: EXECUTIVE SUMMARY ====================
with tab1:
    st.markdown("""
    <div class="section-header">
        <span class="icon">📊</span>
        <span class="text">Business Metrics Overview</span>
    </div>
    """, unsafe_allow_html=True)
    
    total_customers = df_cltv.shape[0]
    churn_rate = (df_cltv['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
    avg_cltv = df_cltv['CLTV'].mean()
    total_projected_value = df_cltv.loc[df_cltv['Churn'] == 'No', 'CLTV'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="glass-card" style="--accent-color: #818cf8;">
            <span class="card-icon">👥</span>
            <div class="card-label">Total Customers</div>
            <div class="card-value" style="color: #818cf8;">{total_customers:,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="glass-card" style="--accent-color: #f87171;">
            <span class="card-icon">⚠️</span>
            <div class="card-label">Churn Rate</div>
            <div class="card-value" style="color: #f87171;">{churn_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="glass-card" style="--accent-color: #34d399;">
            <span class="card-icon">💎</span>
            <div class="card-label">Average CLTV</div>
            <div class="card-value" style="color: #34d399;">${avg_cltv:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="glass-card" style="--accent-color: #38bdf8;">
            <span class="card-icon">🏦</span>
            <div class="card-label">Active Portfolio</div>
            <div class="card-value" style="color: #38bdf8;">${total_projected_value/1e6:.1f}M</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">🔍</span>
        <span class="text">Key Churn Drivers</span>
    </div>
    <div class="section-desc">Visual analysis of the primary factors correlated with customer attrition.</div>
    """, unsafe_allow_html=True)
    
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        contract_churn = df_cltv.groupby(['Contract', 'Churn']).size().reset_index(name='Count')
        fig_contract = px.bar(
            contract_churn, x='Contract', y='Count', color='Churn',
            title='Churn by Contract Type',
            barmode='group',
            color_discrete_map=COLOR_MAP_CHURN,
        )
        fig_contract.update_layout(**CHART_LAYOUT)
        fig_contract.update_traces(marker_line_width=0, opacity=0.9)
        st.plotly_chart(fig_contract, use_container_width=True)
        
    with col_plot2:
        fig_charges = px.box(
            df_cltv, x='Churn', y='MonthlyCharges',
            color='Churn',
            title='Monthly Charges by Churn Status',
            color_discrete_map=COLOR_MAP_CHURN,
        )
        fig_charges.update_layout(**CHART_LAYOUT)
        st.plotly_chart(fig_charges, use_container_width=True)
        
    payment_churn = df_cltv.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='Count')
    fig_payment = px.bar(
        payment_churn, y='PaymentMethod', x='Count', color='Churn',
        title='Churn Distribution by Payment Method',
        barmode='stack',
        orientation='h',
        color_discrete_map=COLOR_MAP_CHURN,
        height=380
    )
    fig_payment.update_layout(**CHART_LAYOUT)
    fig_payment.update_traces(marker_line_width=0, opacity=0.9)
    st.plotly_chart(fig_payment, use_container_width=True)

# ==================== TAB 2: CHURN RISK SIMULATOR ====================
with tab2:
    st.markdown("""
    <div class="section-header">
        <span class="icon">🎯</span>
        <span class="text">Real-Time Customer Risk Assessment</span>
    </div>
    <div class="section-desc">Configure customer attributes to simulate churn probability, survival-based CLTV, AI-driven explanations, and prescriptive next-best-actions.</div>
    """, unsafe_allow_html=True)
    
    col_sim_in, col_sim_out = st.columns([1, 1], gap="large")
    
    with col_sim_in:
        st.markdown("""
        <div class="section-header" style="margin-top:0;">
            <span class="icon">📝</span>
            <span class="text">Customer Profile</span>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            sim_gender = st.selectbox("Gender", ["Female", "Male"])
            sim_senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            sim_partner = st.selectbox("Partner", ["No", "Yes"])
        with c2:
            sim_dependents = st.selectbox("Dependents", ["No", "Yes"])
            sim_phone = st.selectbox("Phone Service", ["Yes", "No"])
            if sim_phone == "Yes":
                sim_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
            else:
                sim_lines = "No phone service"
        
        sim_tenure = st.slider("Tenure (Months)", 1, 72, 12)
        sim_internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        
        if sim_internet != "No":
            c3, c4 = st.columns(2)
            with c3:
                sim_sec = st.selectbox("Online Security", ["No", "Yes"])
                sim_back = st.selectbox("Online Backup", ["No", "Yes"])
                sim_prot = st.selectbox("Device Protection", ["No", "Yes"])
            with c4:
                sim_support = st.selectbox("Tech Support", ["No", "Yes"])
                sim_tv = st.selectbox("Streaming TV", ["No", "Yes"])
                sim_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        else:
            sim_sec = sim_back = sim_prot = sim_support = sim_tv = sim_movies = "No internet service"
            
        sim_contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        sim_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        sim_payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        sim_monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0)
        sim_total = sim_monthly * sim_tenure
        
    with col_sim_out:
        st.markdown("""
        <div class="section-header" style="margin-top:0;">
            <span class="icon">⚡</span>
            <span class="text">Risk Engine Output</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Build feature vector
        input_dict = {feat: 0 for feat in features_list}
        num_df = pd.DataFrame([[sim_tenure, sim_monthly, sim_total]], columns=config.NUMERIC_COLS)
        scaled_nums = scaler.transform(num_df)[0]
        input_dict['tenure'] = scaled_nums[0]
        input_dict['MonthlyCharges'] = scaled_nums[1]
        input_dict['TotalCharges'] = scaled_nums[2]
        
        if sim_gender == "Male": input_dict['gender_Male'] = 1
        if sim_senior == "Yes": input_dict['SeniorCitizen_Yes'] = 1
        if sim_partner == "Yes": input_dict['Partner_Yes'] = 1
        if sim_dependents == "Yes": input_dict['Dependents_Yes'] = 1
        if sim_phone == "Yes": input_dict['PhoneService_Yes'] = 1
        if sim_lines == "No phone service": input_dict['MultipleLines_No phone service'] = 1
        elif sim_lines == "Yes": input_dict['MultipleLines_Yes'] = 1
        if sim_internet == "Fiber optic": input_dict['InternetService_Fiber optic'] = 1
        elif sim_internet == "No": input_dict['InternetService_No'] = 1
        
        for name, val in [
            ('OnlineSecurity', sim_sec), ('OnlineBackup', sim_back), 
            ('DeviceProtection', sim_prot), ('TechSupport', sim_support), 
            ('StreamingTV', sim_tv), ('StreamingMovies', sim_movies)
        ]:
            if val == "No internet service": input_dict[f'{name}_No internet service'] = 1
            elif val == "Yes": input_dict[f'{name}_Yes'] = 1
                
        if sim_contract == "One year": input_dict['Contract_One year'] = 1
        elif sim_contract == "Two year": input_dict['Contract_Two year'] = 1
        if sim_billing == "Yes": input_dict['PaperlessBilling_Yes'] = 1
        if sim_payment == "Credit card (automatic)": input_dict['PaymentMethod_Credit card (automatic)'] = 1
        elif sim_payment == "Electronic check": input_dict['PaymentMethod_Electronic check'] = 1
        elif sim_payment == "Mailed check": input_dict['PaymentMethod_Mailed check'] = 1
            
        sim_df = pd.DataFrame([input_dict])
        prob = model.predict_proba(sim_df)[0, 1]
        
        # Risk Gauge
        risk_color = "#34d399" if prob < 0.3 else ("#fbbf24" if prob < 0.7 else "#f87171")
        risk_level = "LOW RISK" if prob < 0.3 else ("MEDIUM RISK" if prob < 0.7 else "HIGH RISK")
        risk_bg = "rgba(52,211,153,0.08)" if prob < 0.3 else ("rgba(251,191,36,0.08)" if prob < 0.7 else "rgba(248,113,113,0.08)")
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            number={'suffix': '%', 'font': {'size': 38, 'color': risk_color, 'family': 'JetBrains Mono'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Churn Probability — {risk_level}", 'font': {'size': 14, 'color': '#8b949e', 'family': 'Inter'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'dtick': 25},
                'bar': {'color': risk_color, 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 30], 'color': 'rgba(52, 211, 153, 0.08)'},
                    {'range': [30, 70], 'color': 'rgba(251, 191, 36, 0.08)'},
                    {'range': [70, 100], 'color': 'rgba(248, 113, 113, 0.08)'}
                ],
                'threshold': {
                    'line': {'color': risk_color, 'width': 3},
                    'thickness': 0.8,
                    'value': prob * 100
                }
            }
        ))
        fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': '#c9d1d9'}, height=220, margin=dict(l=30,r=30,t=40,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # CLTV via Cox Survival
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
        customer_survival = cox_model.predict_survival_function(cox_df)
        
        closest_t = min(customer_survival.index, key=lambda x: abs(x - sim_tenure))
        s_current = customer_survival.loc[closest_t].values[0]
        future_times = [t for t in customer_survival.index if t > sim_tenure and t <= 72]
        
        if s_current == 0 or len(future_times) == 0:
            remaining_tenure = 6.0
        else:
            s_future = customer_survival.loc[future_times].iloc[:, 0]
            remaining_tenure = (s_future / s_current).sum()
            
        expected_total_tenure = sim_tenure + remaining_tenure
        
        st.markdown("<br><div style='font-size:0.85rem; color:#8b949e; margin-bottom:-10px;'>Adjust Expected Profit Margin</div>", unsafe_allow_html=True)
        profit_margin = st.slider("Profit Margin (%)", min_value=10, max_value=100, value=70, step=5, label_visibility="collapsed") / 100.0
        
        sim_cltv = expected_total_tenure * sim_monthly * profit_margin
        
        # Metrics Row
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(f"""
            <div class="glass-card" style="--accent-color: #a78bfa; padding: 14px 12px;">
                <div class="card-label" style="font-size:0.65rem;">Remaining Months</div>
                <div class="card-value" style="color: #a78bfa; font-size: 1.4rem;">{remaining_tenure:.0f}</div>
            </div>""", unsafe_allow_html=True)
        with col_c2:
            st.markdown(f"""
            <div class="glass-card" style="--accent-color: #34d399; padding: 14px 12px;">
                <div class="card-label" style="font-size:0.65rem;">Predicted CLTV</div>
                <div class="card-value" style="color: #34d399; font-size: 1.4rem;">${sim_cltv:,.0f}</div>
            </div>""", unsafe_allow_html=True)
        with col_c3:
            st.markdown(f"""
            <div class="glass-card" style="--accent-color: #38bdf8; padding: 14px 12px;">
                <div class="card-label" style="font-size:0.65rem;">Survival at Tenure</div>
                <div class="card-value" style="color: #38bdf8; font-size: 1.4rem;">{s_current*100:.0f}%</div>
            </div>""", unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
            
        # Survival Curve
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=list(customer_survival.index), 
            y=list(customer_survival.iloc[:, 0]), 
            mode='lines', 
            name='Survival Probability',
            line=dict(color='#818cf8', width=2.5),
            fill='tozeroy',
            fillcolor='rgba(129, 140, 248, 0.06)'
        ))
        fig_curve.add_trace(go.Scatter(
            x=[sim_tenure], y=[s_current], 
            mode='markers+text', 
            name='Current State',
            marker=dict(color='#f87171', size=10, symbol='circle', line=dict(width=2, color='#0a0e17')),
            text=[f'{s_current*100:.0f}%'], textposition='top center', textfont=dict(color='#f87171', size=11)
        ))
        fig_curve.update_layout(
            **CHART_LAYOUT,
            title="Individual Survival Curve",
            xaxis_title="Tenure (Months)", yaxis_title="Survival Probability",
            height=250, showlegend=False
        )
        st.plotly_chart(fig_curve, use_container_width=True)
        
        # SHAP Explanation
        st.markdown("""
        <div class="section-header">
            <span class="icon">🧠</span>
            <span class="text">AI Prediction Explanation (SHAP)</span>
        </div>
        """, unsafe_allow_html=True)
        
        shap_values = shap_explainer(sim_df)
        contributions = shap_values.values[0]
        if len(contributions.shape) > 1:
            contributions = contributions[:, 1]
            
        shap_df = pd.DataFrame({'Feature': list(sim_df.columns), 'Impact': contributions})
        shap_df['Abs_Impact'] = shap_df['Impact'].abs()
        top_shap = shap_df[shap_df['Abs_Impact'] > 0.05].sort_values(by='Abs_Impact', ascending=False).head(6)
        top_shap = top_shap.sort_values(by='Impact', ascending=True)
        
        fig_shap = go.Figure(go.Bar(
            x=top_shap['Impact'],
            y=top_shap['Feature'],
            orientation='h',
            marker_color=top_shap['Impact'].apply(lambda x: '#f87171' if x > 0 else '#34d399'),
            marker_line_width=0,
            text=top_shap['Impact'].apply(lambda x: f"{x:+.3f}"),
            textposition='auto',
            textfont=dict(family='JetBrains Mono', size=11)
        ))
        fig_shap.update_layout(**CHART_LAYOUT)
        fig_shap.update_layout(
            title="Feature Impact on Churn Risk (Log-Odds)",
            xaxis_title="← Reduces Risk          Increases Risk →",
            yaxis_title="",
            height=280,
            margin=dict(l=10, r=10, t=50, b=30)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        
        # Next Best Action Engine
        st.markdown("""
        <div class="section-header">
            <span class="icon">🚀</span>
            <span class="text">Prescriptive Next Best Action</span>
        </div>
        """, unsafe_allow_html=True)
        
        positive_drivers = top_shap[top_shap['Impact'] > 0].sort_values('Impact', ascending=False)
        
        if prob < 0.3:
            st.markdown("""
            <div class="nba-card" style="border-color: rgba(52,211,153,0.3);">
                <div class="nba-label" style="color: #34d399;">✅ STATUS: SECURE</div>
                <div class="nba-text">This customer profile shows <strong>low churn risk</strong>. 
                <strong>Recommended:</strong> Cross-sell premium add-on services and enroll in the referral rewards program to maximize CLTV.</div>
            </div>
            """, unsafe_allow_html=True)
        elif len(positive_drivers) == 0:
            st.markdown("""
            <div class="nba-card" style="border-color: rgba(251,191,36,0.3);">
                <div class="nba-label" style="color: #fbbf24;">⚠️ STATUS: MIXED SIGNALS</div>
                <div class="nba-text">No single dominant risk driver identified. 
                <strong>Recommended:</strong> Schedule a proactive customer success check-in call to assess satisfaction.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            top_driver = positive_drivers.iloc[0]['Feature']
            
            if 'Contract' in top_driver:
                nba_icon = "📋"; nba = "Offer a <strong>10-20% discount</strong> to lock in a 1-year or 2-year contract upgrade."
            elif 'Fiber optic' in top_driver:
                nba_icon = "🌐"; nba = "Price sensitivity detected. <strong>Check service quality</strong>, investigate outages, and offer a temporary price freeze."
            elif 'TechSupport' in top_driver or 'OnlineSecurity' in top_driver:
                nba_icon = "🛡️"; nba = "Offer a <strong>3-month complimentary trial</strong> of Premium Tech Support & Online Security bundle."
            elif 'MonthlyCharges' in top_driver:
                nba_icon = "💰"; nba = "Review data usage patterns and recommend a <strong>more cost-effective plan</strong> to improve perceived value."
            elif 'PaymentMethod' in top_driver:
                nba_icon = "💳"; nba = "Offer a <strong>$5/month credit</strong> for switching to Automatic Credit Card payments."
            elif 'tenure' in top_driver:
                nba_icon = "🆕"; nba = "Early lifecycle risk. Trigger the <strong>personalized Welcome & Onboarding</strong> nurture sequence."
            else:
                nba_icon = "🎯"; nba = f"Address the <strong>{top_driver}</strong> concern with a targeted customer outreach follow-up."
                
            st.markdown(f"""
            <div class="nba-card high-risk" style="border-color: rgba(248,113,113,0.3);">
                <div class="nba-label" style="color: #f87171;">🚨 PRIMARY CHURN DRIVER: {top_driver}</div>
                <div class="nba-text">{nba_icon} {nba}</div>
            </div>
            """, unsafe_allow_html=True)

# ==================== TAB 3: CUSTOMER SEGMENTATION ====================
with tab3:
    st.markdown("""
    <div class="section-header">
        <span class="icon">👥</span>
        <span class="text">Behavioral Customer Segmentation</span>
    </div>
    <div class="section-desc">K-Means clustering (K=4) on engagement signals — tenure, monthly charges, and total charges — reveals four distinct customer archetypes with unique retention strategies.</div>
    """, unsafe_allow_html=True)
    
    col_seg1, col_seg2 = st.columns(2)
    with col_seg1:
        st.markdown("""
        <div class="strat-card" style="--card-accent: #34d399;">
            <h4 style="color:#34d399;margin:0 0 6px 0;">🏆 Loyal Premium</h4>
            <p>High tenure, high monthly spend. VIP subscribers with multiple add-ons. 
            <b>Strategy:</b> Premium cross-selling (streaming, home protection) and exclusive loyalty benefits.</p>
        </div>
        <div class="strat-card" style="--card-accent: #f87171;">
            <h4 style="color:#f87171;margin:0 0 6px 0;">🔥 High-Spend At-Risk</h4>
            <p>Low tenure, high monthly charges. Typically new fiber-optic signups on month-to-month contracts. 
            <b>Strategy:</b> Proactive support outreach, contract upgrade discounts, lock-in pricing offers.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_seg2:
        st.markdown("""
        <div class="strat-card" style="--card-accent: #38bdf8;">
            <h4 style="color:#38bdf8;margin:0 0 6px 0;">💙 Loyal Value</h4>
            <p>High tenure, low monthly spend. Budget-conscious subscribers with long-term contracts. 
            <b>Strategy:</b> Offer minor upgrades, reward longevity, contract roll-overs.</p>
        </div>
        <div class="strat-card" style="--card-accent: #fbbf24;">
            <h4 style="color:#fbbf24;margin:0 0 6px 0;">🌱 New Budget</h4>
            <p>Low tenure, low monthly spend. New trial-like subscribers. 
            <b>Strategy:</b> Frictionless digital onboarding, utility tutorials, engagement campaigns.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">📋</span>
        <span class="text">Segment Metrics Profiles</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.dataframe(pd.DataFrame(segment_profiles).rename(columns={
        'Count': 'Customer Count',
        'Avg_Tenure': 'Avg Tenure (m)',
        'Avg_MonthlyCharges': 'Avg Monthly ($)',
        'Avg_TotalCharges': 'Avg Total ($)',
        'Churn_Rate': 'Churn Rate (%)',
        'Strategic_Recommendation': 'Retention Action'
    }), use_container_width=True, hide_index=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">🌐</span>
        <span class="text">Interactive 3D Cluster Space</span>
    </div>
    <div class="section-desc">PCA projection of scaled engagement variables into 3D space for high-fidelity visualization of cluster separation.</div>
    """, unsafe_allow_html=True)
    
    features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler_pca = StandardScaler()
    scaled_feats = scaler_pca.fit_transform(df_cltv[features])
    pca = PCA(n_components=3, random_state=config.RANDOM_STATE)
    pca_coords = pca.fit_transform(scaled_feats)
    
    df_pca = df_cltv.copy()
    df_pca['PCA 1'] = pca_coords[:, 0]
    df_pca['PCA 2'] = pca_coords[:, 1]
    df_pca['PCA 3'] = pca_coords[:, 2]
    df_pca_sample = df_pca.sample(n=2500, random_state=config.RANDOM_STATE)
    
    fig_pca = px.scatter_3d(
        df_pca_sample, x='PCA 1', y='PCA 2', z='PCA 3',
        color='Segment',
        hover_data=['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn'],
        title="3D Customer Segment Projection",
        color_discrete_map=COLOR_MAP_SEGMENTS,
        height=600
    )
    fig_pca.update_layout(
        **CHART_LAYOUT,
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(99,102,241,0.06)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(99,102,241,0.06)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(99,102,241,0.06)"),
        ),
    )
    fig_pca.update_traces(marker=dict(size=3, opacity=0.75, line=dict(width=0)))
    st.plotly_chart(fig_pca, use_container_width=True)

# ==================== TAB 4: CLTV & SURVIVAL ANALYTICS ====================
with tab4:
    st.markdown("""
    <div class="section-header">
        <span class="icon">📈</span>
        <span class="text">Survival Analysis & Lifetime Value Cohorts</span>
    </div>
    <div class="section-desc">Modeling the probability of customer retention over time using Kaplan-Meier estimation and Cox Proportional Hazards. Longer survival directly drives higher CLTV.</div>
    """, unsafe_allow_html=True)
             
    col_surv_left, col_surv_right = st.columns(2)
    
    with col_surv_left:
        st.markdown("""
        <div class="section-header" style="margin-top:0;">
            <span class="icon">📉</span>
            <span class="text">Kaplan-Meier Curves by Contract</span>
        </div>
        """, unsafe_allow_html=True)
        
        km_colors = {'Month-to-month': '#f87171', 'One year': '#fbbf24', 'Two year': '#34d399'}
        fig_km = go.Figure()
        for contract_type, c_data in km_cohorts_data.items():
            fig_km.add_trace(go.Scatter(
                x=c_data['timeline'],
                y=c_data['survival_probability'],
                mode='lines',
                name=contract_type,
                line=dict(width=2.5, color=km_colors.get(contract_type, '#818cf8'))
            ))
        fig_km.update_layout(**CHART_LAYOUT, height=400, xaxis_title="Tenure (Months)", yaxis_title="Retention Probability")
        st.plotly_chart(fig_km, use_container_width=True)
        
    with col_surv_right:
        st.markdown("""
        <div class="section-header" style="margin-top:0;">
            <span class="icon">💎</span>
            <span class="text">CLTV Distribution</span>
        </div>
        """, unsafe_allow_html=True)
        
        fig_cltv_dist = px.histogram(
            df_cltv, x='CLTV', color='CLTV_Level',
            nbins=50,
            title='Customer Count by Lifetime Value ($)',
            color_discrete_map={'Low Value': '#f87171', 'Medium Value': '#fbbf24', 'High Value': '#34d399'},
        )
        fig_cltv_dist.update_layout(**CHART_LAYOUT, height=400)
        fig_cltv_dist.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig_cltv_dist, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">🎯</span>
        <span class="text">Risk vs. Lifetime Value Strategy Matrix</span>
    </div>
    <div class="section-desc">Crossing ML-predicted churn probability with survival-based CLTV to create four strategic action buckets for optimal budget allocation.</div>
    """, unsafe_allow_html=True)
             
    try:
        df_ml = pd.read_csv(config.PREPROCESSED_DATA_PATH)
        X_all = df_ml.drop(config.TARGET_COL, axis=1)
        churn_probs = model.predict_proba(X_all)[:, 1]
    except:
        churn_probs = np.where(df_cltv['Churn'] == 'Yes', 0.8, 0.1)
        
    df_matrix = df_cltv.copy()
    df_matrix['Churn_Prob'] = churn_probs
    median_cltv = df_matrix['CLTV'].median()
    
    def assign_bucket(row):
        is_high_risk = row['Churn_Prob'] >= 0.4
        is_high_val = row['CLTV'] >= median_cltv
        if is_high_risk and is_high_val: return "VIP at Risk"
        elif not is_high_risk and is_high_val: return "VIP Loyal"
        elif is_high_risk and not is_high_val: return "Low-Value Churner"
        else: return "Stable Budget"
            
    df_matrix['Action_Bucket'] = df_matrix.apply(assign_bucket, axis=1)
    bucket_counts = df_matrix['Action_Bucket'].value_counts()
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        fig_pie = px.pie(
            values=bucket_counts.values,
            names=bucket_counts.index,
            hole=0.45,
            color=bucket_counts.index,
            color_discrete_map={
                'VIP at Risk': '#f87171',
                'VIP Loyal': '#34d399',
                'Low-Value Churner': '#fbbf24',
                'Stable Budget': '#38bdf8'
            },
        )
        fig_pie.update_layout(**CHART_LAYOUT, height=400, title="Strategic Bucket Distribution", showlegend=True)
        fig_pie.update_traces(textfont=dict(family='JetBrains Mono', size=12), textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b2:
        st.markdown(f"""
        <div class="strat-card" style="--card-accent: #f87171;">
            <h4 style="color:#f87171;margin:0 0 6px 0;">🚨 VIP at Risk ({bucket_counts.get('VIP at Risk', 0):,})</h4>
            <p>High CLTV, high churn risk. <b>Playbook:</b> Assign dedicated account manager. Maximum contract discount incentives. Priority technical issue resolution.</p>
        </div>
        <div class="strat-card" style="--card-accent: #34d399;">
            <h4 style="color:#34d399;margin:0 0 6px 0;">💚 VIP Loyal ({bucket_counts.get('VIP Loyal', 0):,})</h4>
            <p>High CLTV, low churn risk. <b>Playbook:</b> Enroll in referral programs. Cross-sell premium add-ons. Request reviews and case studies.</p>
        </div>
        <div class="strat-card" style="--card-accent: #fbbf24;">
            <h4 style="color:#fbbf24;margin:0 0 6px 0;">⚡ Low-Value Churner ({bucket_counts.get('Low-Value Churner', 0):,})</h4>
            <p>Low CLTV, high churn risk. <b>Playbook:</b> Automated digital email campaigns with low-cost offers. No manual sales rep hours.</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== TAB 5: BATCH CSV SCORING ====================
with tab5:
    st.markdown("""
    <div class="section-header">
        <span class="icon">🗂️</span>
        <span class="text">Batch Customer Scoring</span>
    </div>
    <div class="section-desc">Upload a raw CSV of customer data to automatically clean, preprocess, score churn risk, and output the results.</div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Raw Customer Data (CSV)", type="csv")
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {batch_df.shape[0]} rows!")
            
            with st.spinner("Processing data and generating predictions..."):
                import sys
                import os
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from src import config
                from src.preprocess import clean_data
                
                # 1. Clean data
                clean_batch = clean_data(batch_df)
                
                # 2. Extract features exactly as model expects
                cat_cols = [c for c in config.CATEGORICAL_COLS if c in clean_batch.columns and c != config.TARGET_COL]
                encode_batch = pd.get_dummies(clean_batch, columns=cat_cols, drop_first=True)
                
                bool_cols = encode_batch.select_dtypes(include='bool').columns
                encode_batch[bool_cols] = encode_batch[bool_cols].astype(int)
                
                num_cols = [c for c in config.NUMERIC_COLS if c in encode_batch.columns]
                encode_batch[num_cols] = scaler.transform(encode_batch[num_cols])
                
                # Ensure all required features are present
                for feat in features_list:
                    if feat not in encode_batch.columns:
                        encode_batch[feat] = 0
                        
                # Order features
                X_batch = encode_batch[features_list]
                
                # 3. Predict
                churn_probs = model.predict_proba(X_batch)[:, 1]
                batch_df['Churn_Risk_Probability'] = churn_probs.round(4)
                batch_df['Risk_Level'] = np.where(churn_probs > 0.7, 'High', np.where(churn_probs > 0.3, 'Medium', 'Low'))
                
                st.markdown("### Scoring Results Preview")
                st.dataframe(batch_df[['customerID', 'tenure', 'MonthlyCharges', 'Churn_Risk_Probability', 'Risk_Level']].head(15), use_container_width=True)
                
                # 4. Download
                csv_out = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Scored CSV",
                    data=csv_out,
                    file_name='scored_customers_batch.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"An error occurred during batch processing: {e}")

# ==================== TAB 6: RFM TRANSACTION ANALYTICS ====================
with tab6:
    st.markdown("""
    <div class="section-header">
        <span class="icon">🛒</span>
        <span class="text">RFM Transaction Analytics</span>
    </div>
    <div class="section-desc">Analysis of historical transaction data to categorize customers based on Recency, Frequency, and Monetary value.</div>
    """, unsafe_allow_html=True)
    
    try:
        df_rfm = pd.read_csv(config.RFM_DATA_PATH)
        
        st.info("💡 **What is RFM?** RFM is an industry-standard behavioral model that ranks customers based on **Recency** (days since last purchase), **Frequency** (number of purchases), and **Monetary** value (total spend). It groups customers into strategic buckets like 'Champions' or 'At Risk' to drive highly targeted retention marketing.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Calculate Averages
        avg_r = int(df_rfm['Recency'].mean())
        avg_f = int(df_rfm['Frequency'].mean())
        avg_m = int(df_rfm['Monetary'].mean())
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
            <div class="glass-card">
                <span class="card-icon">⏳</span>
                <div class="card-label">Avg Recency</div>
                <div class="card-value">{avg_r} <span style="font-size:1rem;color:#8b949e;">days</span></div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="glass-card">
                <span class="card-icon">🔄</span>
                <div class="card-label">Avg Frequency</div>
                <div class="card-value">{avg_f} <span style="font-size:1rem;color:#8b949e;">txns</span></div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class="glass-card">
                <span class="card-icon">💰</span>
                <div class="card-label">Avg Monetary</div>
                <div class="card-value">${avg_m:,}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_r1, col_r2 = st.columns([2, 1])
        
        with col_r1:
            st.markdown("#### RFM Segment Distribution")
            rfm_counts = df_rfm['RFM_Segment'].value_counts().reset_index()
            rfm_counts.columns = ['Segment', 'Count']
            
            fig_donut = px.pie(
                rfm_counts, 
                names='Segment', 
                values='Count',
                hole=0.65,
                title=None,
                color='Segment',
                color_discrete_map={
                    'Champions': '#34d399',
                    'Loyal Customers': '#38bdf8',
                    'Potential Loyalists': '#818cf8',
                    'At Risk': '#fbbf24',
                    'Hibernating': '#f87171',
                    'Needs Attention': '#94a3b8'
                }
            )
            fig_donut.update_traces(
                textinfo='percent+label',
                textposition='outside',
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
                marker=dict(line=dict(color='#0d1321', width=3)),
                textfont_size=13
            )
            fig_donut.update_layout(**CHART_LAYOUT)
            fig_donut.update_layout(
                title='',
                title_text='',
                margin=dict(t=20, l=40, r=40, b=40),
                showlegend=False,
                annotations=[dict(text=f"Total<br><b>{rfm_counts['Count'].sum():,}</b>", x=0.5, y=0.5, font_size=22, showarrow=False, font_color="#e6edf3")]
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with col_r2:
            st.markdown("#### Segment Strategies")
            st.markdown("""
            <div class="strat-card" style="--card-accent: #34d399;">
                <h4 style="color:#34d399;margin:0 0 6px 0;">🏆 Champions</h4>
                <p>Bought recently, buy often, and spend the most. Reward them.</p>
            </div>
            <div class="strat-card" style="--card-accent: #38bdf8;">
                <h4 style="color:#38bdf8;margin:0 0 6px 0;">💚 Loyal Customers</h4>
                <p>Good frequency and recency. Upsell higher-value products.</p>
            </div>
            <div class="strat-card" style="--card-accent: #fbbf24;">
                <h4 style="color:#fbbf24;margin:0 0 6px 0;">⚠️ At Risk</h4>
                <p>Used to buy often, but haven't recently. Send "We miss you" incentives.</p>
            </div>
            <div class="strat-card" style="--card-accent: #f87171;">
                <h4 style="color:#f87171;margin:0 0 6px 0;">❄️ Hibernating</h4>
                <p>Last purchase was long ago. Low marketing priority.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("### RFM Customer Table")
        st.dataframe(df_rfm[['customerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Segment', 'Churn']].head(50), use_container_width=True)
        
    except FileNotFoundError:
        st.warning(f"RFM data not found. Please run `python src/generate_transactions.py` and then `python src/rfm_analysis.py` to generate the synthetic transaction data.")

# ==================== FOOTER ====================
st.markdown("""
<div class="app-footer">
    <strong>ChurnGuard AI</strong> · Built with XGBoost, SHAP, Lifelines & Streamlit · 
    <a href="https://github.com/Anu2030/churn_cltv" target="_blank">GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
