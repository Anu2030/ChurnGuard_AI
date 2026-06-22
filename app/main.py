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
from src import llm_service

# Set page config
st.set_page_config(
    page_title="ChurnGuard AI — Customer Analytics Suite",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PREMIUM DESIGN SYSTEM ====================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
    /* ===== ANIMATIONS ===== */
    @keyframes pulse-risk {
        0% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.5); }
        70% { box-shadow: 0 0 0 12px rgba(244, 63, 94, 0); }
        100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }
    }
    @keyframes pulse-badge {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    @keyframes fade-in-up {
        0% { opacity: 0; transform: translateY(18px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes gradient-sweep {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes top-bar-sweep {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    @keyframes icon-spin-in {
        0% { transform: rotate(-8deg) scale(0.9); opacity: 0; }
        100% { transform: rotate(0deg) scale(1); opacity: 1; }
    }

    /* ===== FUTURISTIC AI CONTAINERS ===== */
    @keyframes ai-glow {
        0% { box-shadow: 0 0 5px rgba(139, 92, 246, 0.1), inset 0 0 10px rgba(20, 184, 166, 0.05); }
        50% { box-shadow: 0 0 20px rgba(139, 92, 246, 0.3), inset 0 0 20px rgba(20, 184, 166, 0.15); }
        100% { box-shadow: 0 0 5px rgba(139, 92, 246, 0.1), inset 0 0 10px rgba(20, 184, 166, 0.05); }
    }
    @keyframes text-fade-in {
        0% { opacity: 0; transform: translateY(5px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    .ai-container {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
        border: 1px solid rgba(139, 92, 246, 0.4);
        border-radius: 14px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 24px;
        animation: ai-glow 4s infinite alternate;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(16px);
    }
    .ai-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #14b8a6, #8b5cf6, #3b82f6, #14b8a6);
        animation: gradient-sweep 3s linear infinite;
        background-size: 200% auto;
    }
    .ai-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #5eead4;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
        font-weight: 600;
        border-bottom: 1px solid rgba(139, 92, 246, 0.2);
        padding-bottom: 12px;
    }
    .ai-header-icon {
        animation: icon-spin-in 1s ease-out;
        font-size: 1.2rem;
    }
    .ai-text {
        font-family: 'Inter', sans-serif;
        color: #f8fafc;
        font-size: 0.95rem;
        line-height: 1.7;
        animation: text-fade-in 1.5s ease-out forwards;
    }

    /* ===== ANIMATED TOP BAR ===== */
    .stApp::before {
        content: '';
        display: block;
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #14b8a6, #8b5cf6, #f43f5e, #f59e0b, #14b8a6);
        background-size: 200% auto;
        animation: top-bar-sweep 4s linear infinite;
        z-index: 9999;
    }

    /* ===== GLOBAL RESET & TYPOGRAPHY ===== */
    *, *::before, *::after { box-sizing: border-box; }
    .stApp {
        background: linear-gradient(160deg, #080c14 0%, #0c1220 45%, #101828 100%);
        color: #cbd5e1;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    div[data-testid="stVerticalBlock"] > div {
        animation: fade-in-up 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #f1f5f9 !important;
        letter-spacing: -0.02em;
    }
    label, p { font-family: 'Inter', sans-serif; }

    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(20, 184, 166, 0.35); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(20, 184, 166, 0.6); }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080c14 0%, #0d1520 100%) !important;
        border-right: 1px solid rgba(20, 184, 166, 0.12) !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.4);
    }
    section[data-testid="stSidebar"]::before {
        content: '';
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: linear-gradient(180deg, #14b8a6, #8b5cf6, #f43f5e);
        border-radius: 0 2px 2px 0;
    }
    section[data-testid="stSidebar"] .stMarkdown p { color: #64748b !important; }

    /* ===== TABS STYLING ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(8, 12, 20, 0.7);
        border-radius: 14px;
        padding: 6px;
        border: 1px solid rgba(20, 184, 166, 0.1);
        backdrop-filter: blur(12px);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 18px;
        font-weight: 500;
        font-size: 0.83rem;
        color: #64748b;
        background: transparent;
        border: none;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
        background: rgba(20, 184, 166, 0.07);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.18), rgba(139, 92, 246, 0.18)) !important;
        color: #5eead4 !important;
        font-weight: 600;
        border: 1px solid rgba(20, 184, 166, 0.25) !important;
        box-shadow: 0 0 18px rgba(20, 184, 166, 0.12);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ===== GLASSMORPHISM METRIC CARDS ===== */
    .glass-card {
        background: linear-gradient(135deg, rgba(12, 18, 32, 0.85), rgba(20, 30, 50, 0.6));
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 18px;
        padding: 26px 22px;
        text-align: center;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .glass-card::after {
        content: '';
        position: absolute;
        bottom: 0; left: 10%; right: 10%;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--accent-color, #14b8a6), transparent);
        border-radius: 0 0 18px 18px;
        opacity: 0.7;
        transition: opacity 0.3s ease;
    }
    .glass-card::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(20,184,166,0.06) 0%, transparent 60%);
        pointer-events: none;
    }
    .glass-card:hover {
        transform: translateY(-7px);
        border-color: rgba(20, 184, 166, 0.25);
        box-shadow: 0 24px 48px rgba(0, 0, 0, 0.4), 0 0 40px rgba(20, 184, 166, 0.08);
    }
    .glass-card:hover::after { opacity: 1; }
    .card-icon {
        font-size: 1.9rem;
        margin-bottom: 10px;
        display: block;
        animation: icon-spin-in 0.5s ease forwards;
    }
    .card-label {
        font-size: 0.68rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }
    .card-value {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ===== SECTION HEADERS ===== */
    .section-header {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 30px 0 18px 0;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(20, 184, 166, 0.1);
        position: relative;
    }
    .section-header::before {
        content: '';
        position: absolute;
        bottom: -1px; left: 0;
        width: 60px; height: 2px;
        background: linear-gradient(90deg, #14b8a6, #8b5cf6);
        border-radius: 2px;
    }
    .section-header .icon {
        font-size: 1.3rem;
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.15), rgba(139, 92, 246, 0.15));
        padding: 9px 11px;
        border-radius: 11px;
        border: 1px solid rgba(20, 184, 166, 0.2);
        transition: transform 0.35s ease, box-shadow 0.35s ease;
        animation: icon-spin-in 0.5s ease forwards;
    }
    .section-header:hover .icon {
        transform: rotate(-6deg) scale(1.08);
        box-shadow: 0 0 20px rgba(20, 184, 166, 0.3);
    }
    .section-header .text {
        font-size: 1.12rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.01em;
    }
    .section-desc {
        color: #64748b;
        font-size: 0.87rem;
        line-height: 1.65;
        margin-bottom: 22px;
    }

    /* ===== STRATEGY CARDS ===== */
    .strat-card {
        background: linear-gradient(135deg, rgba(12, 18, 32, 0.75), rgba(20, 30, 50, 0.45));
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-left: 4px solid var(--card-accent, #14b8a6);
        border-radius: 0 14px 14px 0;
        padding: 18px 20px;
        margin-bottom: 14px;
        transition: all 0.3s ease;
    }
    .strat-card:hover {
        background: linear-gradient(135deg, rgba(12, 18, 32, 0.92), rgba(20, 30, 50, 0.65));
        border-left-width: 6px;
        transform: translateX(7px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.25);
    }
    .strat-card h4 { font-size: 1rem; font-weight: 700; margin: 0 0 6px 0; }
    .strat-card p { font-size: 0.82rem; color: #64748b; line-height: 1.6; margin: 0; }
    .strat-card b { color: #5eead4; }

    /* ===== NBA RESULT CARDS ===== */
    .nba-card {
        background: linear-gradient(135deg, rgba(12, 18, 32, 0.85), rgba(20, 30, 50, 0.55));
        backdrop-filter: blur(18px);
        border: 1px solid rgba(20, 184, 166, 0.15);
        border-radius: 16px;
        padding: 22px 24px;
        margin-top: 12px;
        transition: all 0.3s ease;
    }
    .nba-card.high-risk { animation: pulse-risk 2s infinite; }
    .nba-card:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(0,0,0,0.25); }
    .nba-card .nba-label {
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }
    .nba-card .nba-text { font-size: 0.92rem; line-height: 1.65; color: #cbd5e1; }

    /* ===== MAIN HERO HEADER ===== */
    .hero-wrapper {
        padding: 12px 0 6px 0;
        position: relative;
    }
    .model-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(20, 184, 166, 0.1);
        border: 1px solid rgba(20, 184, 166, 0.25);
        border-radius: 20px;
        padding: 4px 12px 4px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        color: #5eead4;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }
    .model-status-pill .dot {
        width: 7px; height: 7px;
        background: #14b8a6;
        border-radius: 50%;
        animation: pulse-badge 2s ease-in-out infinite;
        box-shadow: 0 0 8px rgba(20, 184, 166, 0.6);
    }
    .hero-header {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        background: linear-gradient(135deg, #5eead4 0%, #a78bfa 35%, #f43f5e 65%, #5eead4 100%);
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px;
        line-height: 1.15;
        animation: gradient-sweep 5s linear infinite;
    }
    .hero-subtitle {
        font-size: 0.87rem;
        color: #475569;
        font-weight: 400;
        margin-bottom: 30px;
        letter-spacing: 0.03em;
        line-height: 1.6;
    }
    .hero-subtitle span {
        color: #5eead4;
        font-weight: 600;
    }

    /* ===== SIDEBAR BRANDING ===== */
    .sidebar-brand {
        text-align: center;
        padding: 24px 10px 26px 10px;
        border-bottom: 1px solid rgba(20, 184, 166, 0.1);
        margin-bottom: 20px;
    }
    .sidebar-logo-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 52px; height: 52px;
        background: linear-gradient(135deg, #14b8a6, #8b5cf6);
        border-radius: 14px;
        font-size: 1.3rem;
        font-weight: 900;
        color: #fff;
        margin-bottom: 10px;
        font-family: 'JetBrains Mono', monospace;
        box-shadow: 0 8px 24px rgba(20, 184, 166, 0.35);
        transition: transform 0.4s ease;
    }
    .sidebar-brand:hover .sidebar-logo-badge { transform: rotate(6deg) scale(1.05); }
    .sidebar-brand .name {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        letter-spacing: -0.01em;
    }
    .sidebar-brand .version {
        font-size: 0.67rem;
        color: #475569;
        font-weight: 500;
        margin-top: 4px;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .sidebar-nav { padding: 8px 0; }
    .sidebar-nav .nav-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 11px 16px;
        border-radius: 10px;
        font-size: 0.84rem;
        font-weight: 500;
        color: #64748b;
        margin-bottom: 4px;
        transition: all 0.2s ease;
        cursor: default;
    }
    .sidebar-nav .nav-item:hover {
        background: rgba(20, 184, 166, 0.1);
        color: #e2e8f0;
        transform: translateX(5px);
    }
    .sidebar-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.64rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    .sidebar-badge.live {
        animation: pulse-badge 2.5s ease-in-out infinite;
    }

    /* ===== FOOTER ===== */
    .app-footer {
        text-align: center;
        padding: 28px 0;
        margin-top: 50px;
        border-top: 1px solid rgba(20, 184, 166, 0.08);
        color: #334155;
        font-size: 0.74rem;
        letter-spacing: 0.04em;
        transition: color 0.3s ease;
    }
    .app-footer:hover { color: #475569; }
    .app-footer a { color: #5eead4; text-decoration: none; transition: color 0.3s ease; }
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
        background: rgba(8, 12, 20, 0.7) !important;
        border-color: rgba(20, 184, 166, 0.15) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(20, 184, 166, 0.4) !important;
        box-shadow: 0 0 12px rgba(20, 184, 166, 0.1);
    }
    .stSlider > div > div > div > div { background-color: #14b8a6 !important; }

    /* ===== DATAFRAME ===== */
    .stDataFrame {
        border: 1px solid rgba(20, 184, 166, 0.1) !important;
        border-radius: 14px !important;
        overflow: hidden;
    }

    /* ===== INFO / ALERT BOXES ===== */
    .stAlert {
        border-radius: 12px !important;
        border-left-color: #14b8a6 !important;
    }

    /* ===== AI BUTTONS ===== */
    div.stButton > button {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.1), rgba(139, 92, 246, 0.1)) !important;
        color: #5eead4 !important;
        border: 1px solid rgba(20, 184, 166, 0.3) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        padding: 10px 18px !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.01em !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.2), rgba(139, 92, 246, 0.2)) !important;
        border-color: rgba(20, 184, 166, 0.6) !important;
        box-shadow: 0 0 18px rgba(20, 184, 166, 0.25), 0 4px 16px rgba(0,0,0,0.3) !important;
        transform: translateY(-2px);
        color: #a78bfa !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 0 8px rgba(20, 184, 166, 0.15) !important;
    }
    div.stDownloadButton > button {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.08), rgba(20, 184, 166, 0.08)) !important;
        color: #38bdf8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        transition: all 0.25s ease !important;
    }
    div.stDownloadButton > button:hover {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.18), rgba(20, 184, 166, 0.15)) !important;
        border-color: rgba(56, 189, 248, 0.6) !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.2) !important;
        transform: translateY(-2px);
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .hero-header { font-size: 1.9rem; }
        .hero-subtitle { font-size: 0.8rem; }
        .glass-card { padding: 16px 14px; }
        .card-value { font-size: 1.55rem; }
        .section-header .text { font-size: 1rem; }
        .strat-card { padding: 14px 16px; margin-bottom: 10px; }
        .nba-card { padding: 16px 18px; }
        .stTabs [data-baseweb="tab"] { padding: 8px 10px; font-size: 0.74rem; }
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
    font=dict(family='Inter, sans-serif', color='#64748b', size=12),
    title_font=dict(size=14, color='#e2e8f0', family='Inter, sans-serif'),
    margin=dict(l=20, r=20, t=50, b=20),
    xaxis=dict(gridcolor='rgba(20, 184, 166, 0.07)', zerolinecolor='rgba(20, 184, 166, 0.12)'),
    yaxis=dict(gridcolor='rgba(20, 184, 166, 0.07)', zerolinecolor='rgba(20, 184, 166, 0.12)'),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=11)),
    colorway=['#14b8a6', '#8b5cf6', '#f43f5e', '#f59e0b', '#38bdf8', '#10b981', '#a78bfa']
)

COLOR_MAP_CHURN = {'Yes': '#f43f5e', 'No': '#14b8a6'}
COLOR_MAP_SEGMENTS = {
    'Loyal Premium': '#14b8a6',
    'Loyal Value': '#38bdf8',
    'High-Spend At-Risk': '#f43f5e',
    'New Budget': '#f59e0b'
}

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo-badge">CG</div>
        <div class="name">ChurnGuard AI</div>
        <div class="version">Enterprise Analytics v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-nav">
        <div class="nav-item">📊 Executive Dashboard</div>
        <div class="nav-item">🎯 Risk Simulator</div>
        <div class="nav-item">👥 Customer Segments</div>
        <div class="nav-item">📈 CLTV &amp; Survival</div>
        <div class="nav-item">🗂️ Batch Scoring</div>
        <div class="nav-item">🛒 RFM Analytics</div>
        <div class="nav-item">💬 Chat with Data</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="padding: 12px 14px;">
        <div style="font-size:0.68rem;font-weight:700;color:#334155;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;">Pipeline Status</div>
        <div style="font-size:0.8rem;color:#64748b;margin-bottom:8px;display:flex;align-items:center;gap:8px;">
            <span class="sidebar-badge live" style="background:rgba(20,184,166,0.15);color:#14b8a6;">● LIVE</span> XGBoost · ROC-AUC 0.844
        </div>
        <div style="font-size:0.8rem;color:#64748b;margin-bottom:8px;display:flex;align-items:center;gap:8px;">
            <span class="sidebar-badge" style="background:rgba(20,184,166,0.1);color:#5eead4;">✓</span> SHAP Explainer Active
        </div>
        <div style="font-size:0.8rem;color:#64748b;margin-bottom:8px;display:flex;align-items:center;gap:8px;">
            <span class="sidebar-badge" style="background:rgba(139,92,246,0.1);color:#a78bfa;">✓</span> K-Means K=4 Fitted
        </div>
        <div style="font-size:0.8rem;color:#64748b;display:flex;align-items:center;gap:8px;">
            <span class="sidebar-badge" style="background:rgba(56,189,248,0.1);color:#38bdf8;">✓</span> Cox PH Model Ready
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

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0 14px; margin-bottom: 10px;">
        <div style="font-size:0.72rem;font-weight:600;color:#8b5cf6;text-transform:uppercase;letter-spacing:1.5px;display:flex;align-items:center;gap:6px;">
            <span>🤖</span> AI Copilot
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    chat_container = st.container(height=280)
    with chat_container:
        if "sidebar_messages" not in st.session_state:
            st.session_state.sidebar_messages = [
                {"role": "assistant", "content": "Hello! I am your ChurnGuard Copilot. How can I help you analyze customer churn or CLTV today?"}
            ]
        for message in st.session_state.sidebar_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
    if side_prompt := st.chat_input("Ask Copilot...", key="sidebar_chat_input"):
        st.session_state.sidebar_messages.append({"role": "user", "content": side_prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(side_prompt)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    context_data = df_cltv.sample(min(100, len(df_cltv))).to_csv(index=False)
                    response = llm_service.chat_with_data(side_prompt, context_data)
                    st.markdown(response)
        st.session_state.sidebar_messages.append({"role": "assistant", "content": response})
        st.rerun()

# ==================== HEADER ====================
st.markdown("""
<div class="hero-wrapper">
    <div class="model-status-pill">
        <span class="dot"></span>
        XGBoost Active &nbsp;|&nbsp; ROC-AUC 0.844 &nbsp;|&nbsp; Recall 79.7%
    </div>
    <div class="hero-header">ChurnGuard Analytics Suite</div>
    <div class="hero-subtitle">
        <span>ML-Powered</span> Churn Prediction &nbsp;·&nbsp; 
        <span>Survival-Based</span> CLTV &nbsp;·&nbsp; 
        <span>Explainable AI</span> (SHAP) &nbsp;·&nbsp; 
        RFM Segmentation
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Executive Summary", 
    "🎯 Churn Risk Simulator", 
    "👥 Customer Segmentation", 
    "📈 CLTV & Survival Analytics",
    "🗂️ Batch CSV Scoring",
    "🛒 RFM Transaction Analytics",
    "🪄 AI Report Generator"
])

# ==================== TAB 1: EXECUTIVE SUMMARY ====================
with tab1:
    total_customers = df_cltv.shape[0]
    churn_rate = (df_cltv['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
    avg_cltv = df_cltv['CLTV'].mean()
    total_projected_value = df_cltv.loc[df_cltv['Churn'] == 'No', 'CLTV'].sum()
    
    metrics_dict = {
        'total_customers': total_customers,
        'churn_rate': round(churn_rate, 1),
        'avg_cltv': round(avg_cltv, 0),
        'total_projected_value': total_projected_value
    }
    
    st.markdown("""
    <div class="section-header" style="margin-top: 0;">
        <span class="icon">🚨</span>
        <span class="text">Predictive Anomaly Feed</span>
    </div>
    """, unsafe_allow_html=True)
    
    alerts = llm_service.generate_anomaly_alerts(metrics_dict)
    alerts_html = ""
    for alert in alerts:
        color = "#f43f5e" if "CRITICAL" in alert else ("#fbbf24" if "WARNING" in alert else "#38bdf8")
        bg_color = "rgba(244, 63, 94, 0.08)" if "CRITICAL" in alert else ("rgba(251, 191, 36, 0.08)" if "WARNING" in alert else "rgba(56, 189, 248, 0.08)")
        border_color = "rgba(244, 63, 94, 0.25)" if "CRITICAL" in alert else ("rgba(251, 191, 36, 0.25)" if "WARNING" in alert else "rgba(56, 189, 248, 0.25)")
        badge = "CRITICAL" if "CRITICAL" in alert else ("WARNING" if "WARNING" in alert else "AI INSIGHT")
        clean_text = alert.replace("CRITICAL ANOMALY: ", "").replace("WARNING: ", "").replace("INSIGHT: ", "")
        
        alerts_html += f'<div style="background: {bg_color}; border: 1px solid {border_color}; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; display: flex; align-items: center; gap: 12px;"><span style="background: {color}; color: #000; font-size: 0.65rem; font-weight: 800; padding: 2px 8px; border-radius: 6px; letter-spacing: 0.5px;">{badge}</span><span style="font-size: 0.88rem; color: #e2e8f0; font-family: \'Inter\', sans-serif;">{clean_text}</span></div>'
    st.markdown(f'<div style="margin-bottom: 20px;">{alerts_html}</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="section-header">
        <span class="icon">📊</span>
        <span class="text">Business Metrics Overview</span>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    metrics_dict = {
        'total_customers': total_customers,
        'churn_rate': round(churn_rate, 1),
        'avg_cltv': round(avg_cltv, 0),
        'total_projected_value': total_projected_value
    }
    with st.spinner("Generating AI Executive Insight..."):
        ai_summary = llm_service.generate_executive_summary(metrics_dict)
    
    st.markdown(f"""
    <div class="ai-container">
        <div class="ai-header">
            <span class="ai-header-icon">✨</span>
            <span>On-Device AI Engine Insight</span>
        </div>
        <div class="ai-text">
            {ai_summary}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
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

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✨ Generate AI Retention Email"):
            with st.spinner("Writing personalized email..."):
                top_shap_features = dict(zip(top_shap['Feature'], top_shap['Impact']))
                customer_profile = {
                    "tenure": sim_tenure,
                    "MonthlyCharges": sim_monthly,
                    "Contract": sim_contract,
                    "InternetService": sim_internet
                }
                email_draft = llm_service.generate_nba_email(customer_profile, prob, top_shap_features)
                
                st.markdown(f"""
                <div class="ai-container">
                    <div class="ai-header">
                        <span class="ai-header-icon">✉️</span>
                        <span>AI Generated Retention Draft</span>
                    </div>
                    <div class="ai-text" style="white-space: pre-wrap;">{email_draft}</div>
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
    
    table_rows = ""
    for profile in segment_profiles:
        seg_name = profile['Segment']
        count = profile['Count']
        tenure = profile['Avg_Tenure']
        monthly = profile['Avg_MonthlyCharges']
        total = profile['Avg_TotalCharges']
        churn = profile['Churn_Rate']
        action = profile['Strategic_Recommendation']
        seg_color = COLOR_MAP_SEGMENTS.get(seg_name, '#e2e8f0')
        
        if churn > 30:
            churn_badge = f'<span style="background: rgba(244, 63, 94, 0.12); color: #f87171; border: 1px solid rgba(244, 63, 94, 0.25); padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; font-family: \'JetBrains Mono\', monospace;">{churn:.1f}%</span>'
        else:
            churn_badge = f'<span style="background: rgba(20, 184, 166, 0.12); color: #34d399; border: 1px solid rgba(20, 184, 166, 0.25); padding: 4px 10px; border-radius: 20px; font-weight: 600; font-size: 0.8rem; font-family: \'JetBrains Mono\', monospace;">{churn:.1f}%</span>'
            
        row = f'<tr style="border-bottom: 1px solid rgba(20, 184, 166, 0.08); background: transparent;">'
        row += f'<td style="padding: 14px 16px; font-weight: 700; color: {seg_color}; font-size: 0.88rem;">{seg_name}</td>'
        row += f'<td style="padding: 14px 16px; text-align: right; color: #e2e8f0; font-family: \'JetBrains Mono\', monospace; font-size: 0.85rem;">{count:,}</td>'
        row += f'<td style="padding: 14px 16px; text-align: right; color: #e2e8f0; font-family: \'JetBrains Mono\', monospace; font-size: 0.85rem;">{tenure:.1f} m</td>'
        row += f'<td style="padding: 14px 16px; text-align: right; color: #e2e8f0; font-family: \'JetBrains Mono\', monospace; font-size: 0.85rem;">${monthly:,.2f}</td>'
        row += f'<td style="padding: 14px 16px; text-align: right; color: #e2e8f0; font-family: \'JetBrains Mono\', monospace; font-size: 0.85rem;">${total:,.2f}</td>'
        row += f'<td style="padding: 14px 16px; text-align: right;">{churn_badge}</td>'
        row += f'<td style="padding: 14px 16px; color: #94a3b8; font-size: 0.82rem; line-height: 1.5; max-width: 350px;">{action}</td>'
        row += '</tr>'
        table_rows += row
        
    table_html = '<div style="overflow-x: auto; border: 1px solid rgba(20, 184, 166, 0.15); border-radius: 14px; background: linear-gradient(135deg, rgba(8, 12, 20, 0.85), rgba(16, 24, 40, 0.7)); backdrop-filter: blur(24px); box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4); margin-bottom: 25px;">'
    table_html += '<table style="width: 100%; border-collapse: collapse; text-align: left; font-family: \'Inter\', sans-serif;">'
    table_html += '<thead><tr style="border-bottom: 1.5px solid rgba(20, 184, 166, 0.2); background: rgba(20, 184, 166, 0.04);">'
    table_html += '<th style="padding: 14px 16px; font-size: 0.68rem; font-weight: 700; color: #5eead4; text-transform: uppercase; letter-spacing: 1.5px;">Segment</th>'
    table_html += '<th style="padding: 14px 16px; font-size: 0.68rem; font-weight: 700; color: #5eead4; text-transform: uppercase; letter-spacing: 1.5px; text-align: right;">Customer Count</th>'
    table_html += '<th style="padding: 14px 16px; font-size: 0.68rem; font-weight: 700; color: #5eead4; text-transform: uppercase; letter-spacing: 1.5px; text-align: right;">Avg Tenure</th>'
    table_html += '<th style="padding: 14px 16px; font-size: 0.68rem; font-weight: 700; color: #5eead4; text-transform: uppercase; letter-spacing: 1.5px; text-align: right;">Avg Monthly</th>'
    table_html += '<th style="padding: 14px 16px; font-size: 0.68rem; font-weight: 700; color: #5eead4; text-transform: uppercase; letter-spacing: 1.5px; text-align: right;">Avg Total</th>'
    table_html += '<th style="padding: 14px 16px; font-size: 0.68rem; font-weight: 700; color: #5eead4; text-transform: uppercase; letter-spacing: 1.5px; text-align: right;">Churn Rate</th>'
    table_html += '<th style="padding: 14px 16px; font-size: 0.68rem; font-weight: 700; color: #5eead4; text-transform: uppercase; letter-spacing: 1.5px;">Strategic Recommendation</th>'
    table_html += '</tr></thead><tbody>' + table_rows + '</tbody></table></div>'
    
    st.markdown(table_html, unsafe_allow_html=True)
    
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
    
    if "show_insight_demographics" not in st.session_state:
        st.session_state.show_insight_demographics = False
        
    if st.button("🤖 Ask AI to Analyze Segment Clusters", key="btn_seg"):
        st.session_state.show_insight_demographics = not st.session_state.show_insight_demographics
        
    if st.session_state.show_insight_demographics:
        with st.spinner("Analyzing cluster demographics..."):
            insight = llm_service.generate_chart_insight("demographics")
            st.markdown(f"""
            <div class="ai-container" style="margin-top: 10px;">
                <div class="ai-header">
                    <span class="ai-header-icon">✨</span>
                    <span>AI Cluster Demographics Interpretation</span>
                </div>
                <div class="ai-text">
                    {insight}
                </div>
            </div>
            """, unsafe_allow_html=True)

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
        
        if "show_insight_services" not in st.session_state:
            st.session_state.show_insight_services = False
            
        if st.button("🤖 Ask AI to Analyze Survival Curves", key="btn_surv_km"):
            st.session_state.show_insight_services = not st.session_state.show_insight_services
            
        if st.session_state.show_insight_services:
            with st.spinner("Analyzing survival curves..."):
                insight = llm_service.generate_chart_insight("services")
                st.markdown(f"""
                <div class="ai-container" style="margin-top: 10px;">
                    <div class="ai-header">
                        <span class="ai-header-icon">✨</span>
                        <span>AI Survival Curve Interpretation</span>
                    </div>
                    <div class="ai-text">
                        {insight}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
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
            color_discrete_map={'Low Value': '#f43f5e', 'Medium Value': '#f59e0b', 'High Value': '#14b8a6'},
        )
        fig_cltv_dist.update_layout(**CHART_LAYOUT, height=400)
        fig_cltv_dist.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig_cltv_dist, use_container_width=True)
        
        if "show_insight_financials" not in st.session_state:
            st.session_state.show_insight_financials = False
            
        if st.button("🤖 Ask AI to Analyze CLTV Distribution", key="btn_cltv_dist"):
            st.session_state.show_insight_financials = not st.session_state.show_insight_financials
            
        if st.session_state.show_insight_financials:
            with st.spinner("Analyzing CLTV distributions..."):
                insight = llm_service.generate_chart_insight("financials")
                st.markdown(f"""
                <div class="ai-container" style="margin-top: 10px;">
                    <div class="ai-header">
                        <span class="ai-header-icon">✨</span>
                        <span>AI CLTV Distribution Interpretation</span>
                    </div>
                    <div class="ai-text">
                        {insight}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
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
                'VIP at Risk': '#f43f5e',
                'VIP Loyal': '#14b8a6',
                'Low-Value Churner': '#f59e0b',
                'Stable Budget': '#38bdf8'
            },
        )
        fig_pie.update_layout(**CHART_LAYOUT, height=400, title="Strategic Bucket Distribution", showlegend=True)
        fig_pie.update_traces(textfont=dict(family='JetBrains Mono', size=12), textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_b2:
        st.markdown(f"""
        <div class="strat-card" style="--card-accent: #f43f5e;">
            <h4 style="color:#f43f5e;margin:0 0 6px 0;">🚨 VIP at Risk ({bucket_counts.get('VIP at Risk', 0):,})</h4>
            <p>High CLTV, high churn risk. <b>Playbook:</b> Assign dedicated account manager. Maximum contract discount incentives. Priority technical issue resolution.</p>
        </div>
        <div class="strat-card" style="--card-accent: #14b8a6;">
            <h4 style="color:#14b8a6;margin:0 0 6px 0;">💎 VIP Loyal ({bucket_counts.get('VIP Loyal', 0):,})</h4>
            <p>High CLTV, low churn risk. <b>Playbook:</b> Enroll in referral programs. Cross-sell premium add-ons. Request reviews and case studies.</p>
        </div>
        <div class="strat-card" style="--card-accent: #f59e0b;">
            <h4 style="color:#f59e0b;margin:0 0 6px 0;">⚡ Low-Value Churner ({bucket_counts.get('Low-Value Churner', 0):,})</h4>
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
                
                st.markdown("""
                <div class="section-header" style="margin-top: 18px;">
                    <span class="icon">📋</span>
                    <span class="text">Scoring Results Preview</span>
                </div>
                """, unsafe_allow_html=True)
                preview_df = batch_df[['customerID', 'tenure', 'MonthlyCharges', 'Churn_Risk_Probability', 'Risk_Level']].head(15)
                batch_rows = ""
                for _, brow in preview_df.iterrows():
                    rl = brow['Risk_Level']
                    if rl == 'High':
                        risk_html = '<span style="background: rgba(244,63,94,0.12); color:#f87171; border:1px solid rgba(244,63,94,0.3); padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:700;">High</span>'
                        prob_color = '#f87171'
                    elif rl == 'Medium':
                        risk_html = '<span style="background: rgba(251,191,36,0.12); color:#fbbf24; border:1px solid rgba(251,191,36,0.3); padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:700;">Medium</span>'
                        prob_color = '#fbbf24'
                    else:
                        risk_html = '<span style="background: rgba(20,184,166,0.12); color:#34d399; border:1px solid rgba(20,184,166,0.3); padding:3px 10px; border-radius:20px; font-size:0.78rem; font-weight:700;">Low</span>'
                        prob_color = '#34d399'
                    batch_rows += f'<tr style="border-bottom: 1px solid rgba(20,184,166,0.07);"><td style="padding:11px 14px; color:#94a3b8; font-size:0.8rem; font-family:JetBrains Mono,monospace;">{brow["customerID"]}</td><td style="padding:11px 14px; text-align:right; color:#e2e8f0; font-family:JetBrains Mono,monospace; font-size:0.83rem;">{int(brow["tenure"])} m</td><td style="padding:11px 14px; text-align:right; color:#e2e8f0; font-family:JetBrains Mono,monospace; font-size:0.83rem;">${brow["MonthlyCharges"]:.2f}</td><td style="padding:11px 14px; text-align:right; color:{prob_color}; font-family:JetBrains Mono,monospace; font-size:0.83rem; font-weight:600;">{brow["Churn_Risk_Probability"]:.3f}</td><td style="padding:11px 14px; text-align:center;">{risk_html}</td></tr>'
                batch_table = '<div style="overflow-x:auto; border:1px solid rgba(20,184,166,0.15); border-radius:14px; background:linear-gradient(135deg,rgba(8,12,20,0.85),rgba(16,24,40,0.7)); backdrop-filter:blur(20px); box-shadow:0 12px 36px rgba(0,0,0,0.35); margin-bottom:20px;"><table style="width:100%; border-collapse:collapse; font-family:Inter,sans-serif;"><thead><tr style="border-bottom:1.5px solid rgba(20,184,166,0.2); background:rgba(20,184,166,0.04);"><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px;">Customer ID</th><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:right;">Tenure</th><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:right;">Monthly Charges</th><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:right;">Churn Probability</th><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:center;">Risk Level</th></tr></thead><tbody>' + batch_rows + '</tbody></table></div>'
                st.markdown(batch_table, unsafe_allow_html=True)
                
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
                    'Champions': '#14b8a6',
                    'Loyal Customers': '#38bdf8',
                    'Potential Loyalists': '#8b5cf6',
                    'At Risk': '#f59e0b',
                    'Hibernating': '#f43f5e',
                    'Needs Attention': '#64748b'
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
            <div class="strat-card" style="--card-accent: #14b8a6;">
                <h4 style="color:#14b8a6;margin:0 0 6px 0;">🏆 Champions</h4>
                <p>Bought recently, buy often, and spend the most. <b>Reward and retain</b> them with loyalty perks.</p>
            </div>
            <div class="strat-card" style="--card-accent: #38bdf8;">
                <h4 style="color:#38bdf8;margin:0 0 6px 0;">💙 Loyal Customers</h4>
                <p>Good frequency and recency. <b>Upsell</b> higher-value products and services.</p>
            </div>
            <div class="strat-card" style="--card-accent: #f59e0b;">
                <h4 style="color:#f59e0b;margin:0 0 6px 0;">⚠️ At Risk</h4>
                <p>Used to buy often, but haven't recently. <b>Send win-back</b> incentives immediately.</p>
            </div>
            <div class="strat-card" style="--card-accent: #f43f5e;">
                <h4 style="color:#f43f5e;margin:0 0 6px 0;">❄️ Hibernating</h4>
                <p>Last purchase was long ago. <b>Low priority</b> — automated re-engagement only.</p>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        <div class="section-header">
            <span class="icon">📋</span>
            <span class="text">RFM Customer Sample</span>
        </div>
        """, unsafe_allow_html=True)
        rfm_display = df_rfm[['customerID', 'Recency', 'Frequency', 'Monetary', 'RFM_Segment', 'Churn']].head(50)
        rfm_seg_colors = {'Champions': '#14b8a6', 'Loyal Customers': '#38bdf8', 'Potential Loyalists': '#8b5cf6', 'At Risk': '#f59e0b', 'Hibernating': '#f43f5e', 'Needs Attention': '#64748b'}
        rfm_rows = ""
        for _, rrow in rfm_display.iterrows():
            seg_c = rfm_seg_colors.get(rrow['RFM_Segment'], '#94a3b8')
            churn_c = '#f87171' if rrow['Churn'] == 'Yes' else '#34d399'
            rfm_rows += f'<tr style="border-bottom:1px solid rgba(20,184,166,0.07);"><td style="padding:11px 14px; color:#94a3b8; font-size:0.8rem; font-family:JetBrains Mono,monospace;">{rrow["customerID"]}</td><td style="padding:11px 14px; text-align:right; color:#e2e8f0; font-family:JetBrains Mono,monospace; font-size:0.83rem;">{int(rrow["Recency"])} d</td><td style="padding:11px 14px; text-align:right; color:#e2e8f0; font-family:JetBrains Mono,monospace; font-size:0.83rem;">{int(rrow["Frequency"])}</td><td style="padding:11px 14px; text-align:right; color:#e2e8f0; font-family:JetBrains Mono,monospace; font-size:0.83rem;">${rrow["Monetary"]:,.0f}</td><td style="padding:11px 14px; font-size:0.83rem; font-weight:700; color:{seg_c};">{rrow["RFM_Segment"]}</td><td style="padding:11px 14px; text-align:center; font-size:0.8rem; font-weight:700; color:{churn_c};">{rrow["Churn"]}</td></tr>'
        rfm_table = '<div style="overflow-x:auto; border:1px solid rgba(20,184,166,0.15); border-radius:14px; background:linear-gradient(135deg,rgba(8,12,20,0.85),rgba(16,24,40,0.7)); backdrop-filter:blur(20px); box-shadow:0 12px 36px rgba(0,0,0,0.35); margin-bottom:20px; max-height:480px; overflow-y:auto;"><table style="width:100%; border-collapse:collapse; font-family:Inter,sans-serif;"><thead><tr style="border-bottom:1.5px solid rgba(20,184,166,0.2); background:rgba(20,184,166,0.04);"><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px;">Customer ID</th><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:right;">Recency</th><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:right;">Frequency</th><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:right;">Monetary</th><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px;">RFM Segment</th><th style="padding:12px 14px; font-size:0.65rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:center;">Churn</th></tr></thead><tbody>' + rfm_rows + '</tbody></table></div>'
        st.markdown(rfm_table, unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.warning(f"RFM data not found. Please run `python src/generate_transactions.py` and then `python src/rfm_analysis.py` to generate the synthetic transaction data.")

# ==================== TAB 7: AI REPORT GENERATOR ====================
with tab7:
    st.markdown("""
    <div class="section-header">
        <span class="icon">🪄</span>
        <span class="text">AI Dynamic Report Generator</span>
    </div>
    <div class="section-desc">Type a query in natural language to dynamically slice the customer dataset and generate AI-driven custom reports, metrics, and visualizations.</div>
    """, unsafe_allow_html=True)
    
    st.markdown("💡 **Suggested Queries:**")
    col_chip1, col_chip2, col_chip3, col_chip4 = st.columns(4)
    suggested_query = ""
    with col_chip1:
        if st.button("🌐 Fiber Optic Segment", key="chip_fiber", use_container_width=True):
            suggested_query = "Show me a report on Fiber Optic users"
    with col_chip2:
        if st.button("📋 Month-to-Month Contracts", key="chip_m2m", use_container_width=True):
            suggested_query = "Month-to-month contracts risk analysis"
    with col_chip3:
        if st.button("👵 Senior Citizens", key="chip_senior", use_container_width=True):
            suggested_query = "Senior citizen demographics report"
    with col_chip4:
        if st.button("🚨 High Churn Risk", key="chip_churn", use_container_width=True):
            suggested_query = "Show me historical churned customers"
            
    query_input = st.text_input(
        "Enter your custom report query:", 
        value=suggested_query if suggested_query else "Show me a report on Fiber Optic users",
        placeholder="e.g., Show me fiber optic users with month-to-month contracts",
        key="report_query_input"
    )
    
    if query_input:
        with st.spinner("AI parsing query and compiling report..."):
            filters, report_title = llm_service.parse_dynamic_report_query(query_input)
            
            filtered_df = df_cltv.copy()
            for col, val in filters.items():
                if col in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df[col] == val]
                    
            st.markdown(f"### {report_title}")
            
            if filtered_df.empty:
                st.warning("No customers match the generated filters. Try broadening your query.")
            else:
                sub_total = len(filtered_df)
                sub_churn_rate = (filtered_df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100)
                sub_avg_cltv = filtered_df['CLTV'].mean()
                sub_total_value = filtered_df.loc[filtered_df['Churn'] == 'No', 'CLTV'].sum()
                
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(f"""
                    <div class="glass-card" style="--accent-color: #818cf8; padding: 16px 14px;">
                        <div class="card-label" style="font-size:0.65rem;">Segment Customers</div>
                        <div class="card-value" style="color: #818cf8; font-size: 1.6rem;">{sub_total:,}</div>
                    </div>""", unsafe_allow_html=True)
                with m2:
                    st.markdown(f"""
                    <div class="glass-card" style="--accent-color: #f87171; padding: 16px 14px;">
                        <div class="card-label" style="font-size:0.65rem;">Segment Churn Rate</div>
                        <div class="card-value" style="color: #f87171; font-size: 1.6rem;">{sub_churn_rate:.1f}%</div>
                    </div>""", unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                    <div class="glass-card" style="--accent-color: #34d399; padding: 16px 14px;">
                        <div class="card-label" style="font-size:0.65rem;">Segment Avg CLTV</div>
                        <div class="card-value" style="color: #34d399; font-size: 1.6rem;">${sub_avg_cltv:,.0f}</div>
                    </div>""", unsafe_allow_html=True)
                with m4:
                    st.markdown(f"""
                    <div class="glass-card" style="--accent-color: #38bdf8; padding: 16px 14px;">
                        <div class="card-label" style="font-size:0.65rem;">Segment Value</div>
                        <div class="card-value" style="color: #38bdf8; font-size: 1.6rem;">${sub_total_value/1e6:.2f}M</div>
                    </div>""", unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                col_chart, col_table = st.columns([1, 1], gap="medium")
                with col_chart:
                    st.markdown("#### AI Visual Synthesis")
                    if 'Contract' not in filters and len(filtered_df['Contract'].unique()) > 1:
                        fig_dyn = px.histogram(
                            filtered_df, x='Contract', color='Churn',
                            barmode='group',
                            title="Distribution by Contract Type",
                            color_discrete_map=COLOR_MAP_CHURN
                        )
                    else:
                        fig_dyn = px.histogram(
                            filtered_df, x='MonthlyCharges', color='Churn',
                            nbins=20,
                            title="Monthly Charges Distribution",
                            color_discrete_map=COLOR_MAP_CHURN
                        )
                    fig_dyn.update_layout(**CHART_LAYOUT)
                    st.plotly_chart(fig_dyn, use_container_width=True)
                    
                with col_table:
                    st.markdown("""
                    <div class="section-header" style="margin-top:0;">
                        <span class="icon">🗂️</span>
                        <span class="text">AI Compiled Data Slice</span>
                    </div>
                    """, unsafe_allow_html=True)
                    ai_slice = filtered_df[['customerID', 'tenure', 'Contract', 'InternetService', 'MonthlyCharges', 'CLTV', 'Churn']].head(100)
                    ai_rows = ""
                    for _, arow in ai_slice.iterrows():
                        churn_c = '#f87171' if arow['Churn'] == 'Yes' else '#34d399'
                        contract_c = '#f87171' if arow['Contract'] == 'Month-to-month' else ('#fbbf24' if arow['Contract'] == 'One year' else '#34d399')
                        ai_rows += f'<tr style="border-bottom:1px solid rgba(20,184,166,0.07);"><td style="padding:9px 12px; color:#94a3b8; font-size:0.78rem; font-family:JetBrains Mono,monospace;">{arow["customerID"]}</td><td style="padding:9px 12px; text-align:right; color:#e2e8f0; font-family:JetBrains Mono,monospace; font-size:0.8rem;">{int(arow["tenure"])} m</td><td style="padding:9px 12px; font-size:0.8rem; font-weight:600; color:{contract_c};">{arow["Contract"]}</td><td style="padding:9px 12px; font-size:0.8rem; color:#94a3b8;">{arow["InternetService"]}</td><td style="padding:9px 12px; text-align:right; color:#e2e8f0; font-family:JetBrains Mono,monospace; font-size:0.8rem;">${arow["MonthlyCharges"]:.2f}</td><td style="padding:9px 12px; text-align:right; color:#38bdf8; font-family:JetBrains Mono,monospace; font-size:0.8rem; font-weight:600;">${arow["CLTV"]:,.0f}</td><td style="padding:9px 12px; text-align:center; font-size:0.8rem; font-weight:700; color:{churn_c};">{arow["Churn"]}</td></tr>'
                    ai_table = '<div style="overflow-x:auto; border:1px solid rgba(20,184,166,0.15); border-radius:14px; background:linear-gradient(135deg,rgba(8,12,20,0.85),rgba(16,24,40,0.7)); backdrop-filter:blur(20px); box-shadow:0 12px 36px rgba(0,0,0,0.35); max-height:280px; overflow-y:auto;"><table style="width:100%; border-collapse:collapse; font-family:Inter,sans-serif;"><thead style="position:sticky; top:0; z-index:2;"><tr style="border-bottom:1.5px solid rgba(20,184,166,0.2); background:rgba(8,12,20,0.95);"><th style="padding:10px 12px; font-size:0.63rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px;">Customer ID</th><th style="padding:10px 12px; font-size:0.63rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:right;">Tenure</th><th style="padding:10px 12px; font-size:0.63rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px;">Contract</th><th style="padding:10px 12px; font-size:0.63rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px;">Internet</th><th style="padding:10px 12px; font-size:0.63rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:right;">Monthly</th><th style="padding:10px 12px; font-size:0.63rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:right;">CLTV</th><th style="padding:10px 12px; font-size:0.63rem; font-weight:700; color:#5eead4; text-transform:uppercase; letter-spacing:1.5px; text-align:center;">Churn</th></tr></thead><tbody>' + ai_rows + '</tbody></table></div>'
                    st.markdown(ai_table, unsafe_allow_html=True)
                    
                    csv_dyn = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Segment Data (CSV)",
                        data=csv_dyn,
                        file_name='ai_generated_report_segment.csv',
                        mime='text/csv',
                        use_container_width=True
                    )

# ==================== FOOTER ====================
st.markdown("""
<div class="app-footer">
    <strong>ChurnGuard AI</strong> &nbsp;·&nbsp; XGBoost · SHAP · Lifelines · Cox PH · Streamlit &nbsp;·&nbsp;
    <a href="https://github.com/Anu2030/ChurnGuard_AI" target="_blank">GitHub Repository</a>
</div>
""", unsafe_allow_html=True)
