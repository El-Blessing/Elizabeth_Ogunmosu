import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pricer import price_barrier_option
from simulator import simulate_paths
import joblib
import os

st.set_page_config(page_title="Barrier Option Pricing", layout="wide")

st.title("📊 Barrier Option Pricing: Binomial Tree vs ML Surrogate")
st.markdown("Compare exact binomial tree pricing with a fast Machine Learning model")

# ================= LOAD OR TRAIN MODEL =================
@st.cache_resource
def load_ml_model():
    """Load the trained ML model"""
    model_path = 'option_pricing_model.joblib'
    scaler_path = 'feature_scaler.joblib'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    else:
        st.warning("⚠️ Model not found. Training now...")
        with st.spinner("Training ML model (30 seconds)..."):
            # Run training
            import subprocess
            result = subprocess.run(['python', 'train_model.py'], capture_output=True, text=True)
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                st.success("✅ Model trained and loaded!")
                return model, scaler
            else:
                st.error("Training failed. Using binomial tree only.")
                return None, None

# ================= SIDEBAR =================
st.sidebar.header("Contract Parameters")

S0 = st.sidebar.slider("Stock Price (S₀)", 60, 140, 100)
K = st.sidebar.slider("Strike Price (K)", 60, 140, 100)
B = st.sidebar.slider("Barrier (B)", 60, 160, 120)
T = st.sidebar.slider("Time to Maturity (years)", 0.2, 2.0, 1.0, 0.1)
r = st.sidebar.slider("Risk-free Rate (%)", 0, 10, 5) / 100
sigma = st.sidebar.slider("Volatility (%)", 10, 60, 30) / 100
n = st.sidebar.slider("Binomial Steps", 20, 100, 50)
barrier_type = st.sidebar.selectbox("Barrier Type", ["up", "down"])
num_paths = st.sidebar.slider("Simulated Paths", 10, 50, 25)

# Adjust barrier if needed
if barrier_type == "up" and B <= S0:
    B = S0 + 5
if barrier_type == "down" and B >= S0:
    B = S0 - 5

# ================= PRICING =================
# Binomial price
binomial_price = price_barrier_option(S0, K, B, T, r, sigma, n, barrier_type)

# ML prediction
model, scaler = load_ml_model()
ml_price = None
if model and scaler:
    features = np.array([[
        S0 / 100, K / 100, B / 100, T, r * 100, sigma * 100, 1 if barrier_type == "up" else 0
    ]])
    features_scaled = scaler.transform(features)
    ml_price = model.predict(features_scaled)[0]
    ml_price = max(ml_price, 0)

# Simulate paths
paths, knocked_out = simulate_paths(S0, B, T, r, sigma, n, num_paths, barrier_type)
knockout_prob = np.mean(knocked_out)

# ================= DISPLAY =================
st.header("Pricing Results")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Binomial Tree", f"${binomial_price:.4f}")
if ml_price:
    c2.metric("ML Surrogate", f"${ml_price:.4f}")
    error = abs(ml_price - binomial_price) / binomial_price * 100 if binomial_price > 0 else 0
    c3.metric("Prediction Error", f"{error:.1f}%")
else:
    c2.metric("ML Surrogate", "Not available")
c4.metric("Knockout Probability", f"{knockout_prob:.1%}")

# Plot paths
st.header("Simulated Stock Paths")
fig, ax = plt.subplots(figsize=(10, 5))
time = np.linspace(0, T, n + 1)

for i in range(num_paths):
    color = 'red' if knocked_out[i] else 'green'
    ax.plot(time, paths[i], color=color, alpha=0.5, linewidth=0.8)

ax.axhline(B, color='black', linestyle='--', label=f'Barrier (B={B})')
ax.axhline(K, color='orange', linestyle=':', label=f'Strike (K={K})')
ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.caption("🔴 Red = Knocked out | 🟢 Green = Survived")
