import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import base64

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import (
    MODEL_FILE, MODELS_DIR, RAW_DATA_FILE
)
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import create_features, encode_categorical

# Page config MUST be the first Streamlit command
st.set_page_config(page_title="InventoryOS™ | AI Forecasting", layout="wide", initial_sidebar_state="expanded")

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'assets')

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def inject_custom_css():
    bg_path = os.path.join(ASSETS_DIR, 'bg.png')
    
    bg_img = ""
    if os.path.exists(bg_path):
        bg_base64 = get_base64_of_bin_file(bg_path)
        bg_img = f"data:image/png;base64,{bg_base64}"
        
    custom_css = f"""
    <style>
    /* Global Theme & typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    /* Login Page Background */
    .login-bg {{
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: url("{bg_img}");
        background-size: cover;
        background-position: center;
        filter: brightness(0.6);
        z-index: -1;
    }}
    
    /* Glassmorphism Login Container */
    .login-container {{
        background: rgba(10, 30, 20, 0.45);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-top: 1px solid rgba(255, 255, 255, 0.4);
        border-left: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 20px;
        padding: 40px;
        color: white;
        text-align: center;
        box-shadow: 0 15px 35px 0 rgba(0, 0, 0, 0.4);
        animation: fadeIn 1.2s cubic-bezier(0.165, 0.84, 0.44, 1);
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(30px) scale(0.95); }}
        to {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}
    
    /* Green Glow Buttons */
    .stButton>button {{
        background: linear-gradient(135deg, #10b981, #047857);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        width: 100%;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
    }}
    
    .stButton>button:hover {{
        background: linear-gradient(135deg, #34d399, #059669);
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(16, 185, 129, 0.4);
        color: white;
        border: none;
    }}
    
    /* Metric Cards Customization */
    [data-testid="stMetric"] {{
        background: linear-gradient(145deg, #ffffff, #f0fdf4);
        border: 1px solid #d1fae5;
        padding: 20px;
        border-radius: 16px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }}
    
    [data-testid="stMetric"]:hover {{
        transform: translateY(-6px);
        box-shadow: 0 12px 24px rgba(16, 185, 129, 0.15);
        border-color: #34d399;
    }}
    
    /* Text Input Focus State */
    .stTextInput input:focus {{
        border-color: #10b981 !important;
        box-shadow: 0 0 0 1px #10b981 !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_artifacts():
    if not os.path.exists(MODEL_FILE):
        return None, None
    model = joblib.load(MODEL_FILE)
    model_columns = joblib.load(os.path.join(MODELS_DIR, 'model_columns.pkl'))
    return model, model_columns

@st.cache_data
def load_and_prep_data():
    df = load_data(RAW_DATA_FILE)
    df = preprocess_data(df)
    df_features = create_features(df)
    df_encoded = encode_categorical(df_features)
    return df, df_features, df_encoded

def login_page():
    inject_custom_css()
    st.markdown('<div class="login-bg"></div>', unsafe_allow_html=True)
    
    logo_path = os.path.join(ASSETS_DIR, 'logo.png')
    logo_html = ""
    if os.path.exists(logo_path):
        logo_base64 = get_base64_of_bin_file(logo_path)
        logo_html = f'<img src="data:image/png;base64,{logo_base64}" width="110" style="margin-bottom: 10px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">'
        
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown(f"""
            <div class="login-container">
                {logo_html}
                <h1 style='margin-bottom: 0px; font-weight: 700; letter-spacing: -1px; font-size: 2.2rem;'>InventoryOS<span style='color: #34d399;'>.</span></h1>
                <p style='color: #a7f3d0; margin-bottom: 30px; font-weight: 300;'>AI Demand Intelligence</p>
                <div id="form-hook"></div>
            </div>
        """, unsafe_allow_html=True)
        
        # We put the form physically inside the column so it roughly aligns over the container
        # Note: HTML inside markdown doesn't perfectly wrap standard Streamlit widgets unless using components.
        # But visually grouping them in a narrow column over the background works effectively.
        with st.form("login_form"):
            st.text_input("Store Executive Email", placeholder="admin@retailgroup.com")
            st.text_input("Access Key", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Authenticate")
            if submitted:
                st.session_state['logged_in'] = True
                st.rerun()

def dashboard():
    inject_custom_css()
    
    logo_path = os.path.join(ASSETS_DIR, 'logo.png')
    if os.path.exists(logo_path):
        logo_base64 = get_base64_of_bin_file(logo_path)
        st.sidebar.markdown(f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_base64}" width="160" style="border-radius: 16px; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);"></div>', unsafe_allow_html=True)
        
    st.title("📦 Retail Intelligence Hub")
    st.markdown("Monitor historical outflow, activate AI forecasting, and automate your restock pipelines.")
    st.divider()
    
    model, model_columns = load_model_and_artifacts()
    if model is None:
        st.error("Forecast Engine Offline. Models not found in system registry.")
        return
        
    df_raw, df_features, df_encoded = load_and_prep_data()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Engine Parameters")
        st.markdown("Tune the forecasting matrix.")
        
        store_ids = sorted(df_raw['store_id'].unique())
        product_ids = sorted(df_raw['product_id'].unique())
        
        selected_store = st.selectbox("🏬 Location ID", store_ids)
        selected_product = st.selectbox("🏷️ Product SKU", product_ids)
        
        forecast_horizon = st.slider("Projection Horizon (Days)", 1, 30, 7)
        
        st.markdown("---")
        if st.button("End Session", key="logout"):
            st.session_state['logged_in'] = False
            st.rerun()
            
    # Filter data
    mask = (df_features['store_id'] == selected_store) & (df_features['product_id'] == selected_product)
    item_history = df_features[mask].copy()
    
    if item_history.empty:
        st.warning("Data telemetry missing for this combination.")
        return
        
    # UI TABS
    tab1, tab2, tab3 = st.tabs(["📊 Telemetry (Historical)", "🔮 AI Forecast Engine", "📦 Automated Directives"])
    
    with tab1:
        st.subheader(f"Outflow Telemetry")
        chart_data = item_history[['date', 'units_sold']].set_index('date')
        st.line_chart(chart_data, y="units_sold", color="#10b981") # Growth Green
        
        with st.expander("Explore Raw Telemetry Feed"):
            st.dataframe(item_history[['date', 'units_sold', 'price', 'promotion', 'holiday', 'stockout_flag']], use_container_width=True)
            
    # Forecasting Logic Sandbox
    last_date = item_history['date'].max()
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, forecast_horizon + 1)]
    future_preds = []
    
    recent_history = item_history.tail(60).copy()
    
    for date in future_dates:
        new_row = {
            'date': date,
            'store_id': selected_store,
            'product_id': selected_product,
            'on_hand_inventory': 0,
            'units_sold': np.nan,
            'price': recent_history.iloc[-1]['price'],
            'promotion': 0,
            'holiday': 0,
            'lead_time_days': 0,
            'stockout_flag': 0
        }
        
        recent_history = pd.concat([recent_history, pd.DataFrame([new_row])], ignore_index=True)
        
        features_df = create_features(recent_history)
        features_encoded = encode_categorical(features_df)
        
        last_row = features_encoded.iloc[[-1]].copy()
        
        for col in model_columns:
            if col not in last_row.columns:
                last_row[col] = 0
                
        X_future = last_row[model_columns]
        
        pred = model.predict(X_future)[0]
        pred = max(0, pred)
        
        future_preds.append(pred)
        recent_history.iloc[-1, recent_history.columns.get_loc('units_sold')] = pred
        
    with tab2:
        st.subheader(f"Predictive Vector: Next {forecast_horizon} Days")
        
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Target Volume': future_preds
        })
        
        plot_history = item_history.tail(30).set_index('date')[['units_sold']]
        plot_history.rename(columns={'units_sold': 'Known Telemetry'}, inplace=True)
        
        plot_forecast = forecast_df.set_index('Date')[['Target Volume']]
        plot_forecast.rename(columns={'Target Volume': 'AI Projection'}, inplace=True)
        
        combined_chart_data = pd.concat([plot_history, plot_forecast], axis=1)
        st.line_chart(combined_chart_data, color=["#94a3b8", "#10b981"]) 
        
        st.markdown("#### Projection Matrix")
        st.dataframe(forecast_df.style.highlight_max(axis=0, color='#d1fae5', subset=['Target Volume']), use_container_width=True)
        
    with tab3:
        st.subheader("Algorithmic Stock Replenishment")
        total_demand = sum(future_preds)
        safety_stock = np.std(item_history['units_sold']) * 1.65
        reorder_point = total_demand + safety_stock
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label=f"Projected Outflow ({forecast_horizon}D)", value=f"{total_demand:.0f} units")
        with col2:
            st.metric(label="Safety Buffer", value=f"{safety_stock:.0f} units", help="95% confidence variance buffer")
        with col3:
            st.metric(label="Calculated Procurement", value=f"{reorder_point:.0f} units", delta="High Priority", delta_color="normal")
            
        st.success(f"**AI Directive:** To maintain 95% SLA and prevent stock disruptions over the next {forecast_horizon} days, authorize an immediate vendor restock of **{reorder_point:.0f} units**.")

def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        
    if not st.session_state['logged_in']:
        login_page()
    else:
        dashboard()

if __name__ == "__main__":
    main()
