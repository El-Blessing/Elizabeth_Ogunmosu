import streamlit as st
import numpy as np
import time
import matplotlib.pyplot as plt
from pricer import price_barrier_option
from simulator import simulate_paths
from scipy.stats import norm

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings in Streamlit
import tensorflow as tf
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

st.set_page_config(page_title="Barrier Option Pricing: Binomial Tree vs RNN-LSTM", layout="wide")

st.title("📊 Barrier Option Pricing: Binomial Tree vs RNN-LSTM")
st.markdown("""
**Practical Demonstration:** Compare exact binomial tree pricing with a trained RNN-LSTM surrogate model.
The neural network learns the pricing function from thousands of binomial tree calculations.
""")

# ================= LOAD OR TRAIN MODEL =================
@st.cache_resource
def load_or_train_model():
    """Load pre-trained model or train if not exists"""
    model_path = 'barrier_rnn_model.h5'
    
    if os.path.exists(model_path):
        st.info("📦 Loading pre-trained RNN-LSTM model...")
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model
    else:
        st.warning("⚠️ No trained model found. Please run 'python train_rnn.py' first.")
        st.info("""
        To train the model:
        1. Open terminal in your project folder
        2. Run: `python train_rnn.py`
        3. Wait 2-3 minutes for training
        4. Restart this app
        """)
        return None

# ================= SIDEBAR =================
st.sidebar.header("📈 Option Contract Parameters")

col1, col2 = st.sidebar.columns(2)
with col1:
    S0 = st.slider("Stock Price (S₀)", 50.0, 150.0, 100.0, 1.0)
    K = st.slider("Strike Price (K)", 50.0, 150.0, 100.0, 1.0)
    T = st.slider("Time to Maturity (years)", 0.1, 2.0, 1.0, 0.1)

with col2:
    sigma = st.slider("Volatility (σ)", 0.10, 0.60, 0.30, 0.01)
    r = st.slider("Risk-free Rate (r)", 0.0, 0.10, 0.05, 0.01)
    n = st.slider("Binomial Steps", 20, 100, 50, 10)

barrier_type = st.sidebar.selectbox("Barrier Type", ["up-and-out", "down-and-out"])
B = st.sidebar.slider(
    "Barrier Level (B)", 
    50.0, 200.0, 
    120.0 if barrier_type == "up-and-out" else 80.0, 
    1.0
)

num_paths = st.sidebar.slider("Number of Simulated Paths", 10, 50, 25)

# Auto-correct barrier
barrier_short = 'up' if barrier_type == "up-and-out" else 'down'
if barrier_short == 'up' and B <= S0:
    B = S0 + 5
    st.sidebar.warning(f"Barrier adjusted to {B} (must be above S₀ for up-and-out)")
elif barrier_short == 'down' and B >= S0:
    B = S0 - 5
    st.sidebar.warning(f"Barrier adjusted to {B} (must be below S₀ for down-and-out)")

# ================= PRICE CALCULATIONS =================
with st.spinner("Calculating binomial tree price..."):
    binomial_price = price_barrier_option(S0, K, B, T, r, sigma, n, barrier_short)

# RNN Prediction
model = load_or_train_model()
rnn_price = None
rnn_error = None

if model:
    with st.spinner("Generating RNN-LSTM prediction..."):
        # Generate a path for current parameters
        paths, _ = simulate_paths(S0, B, T, r, sigma, n, 1, barrier_short)
        norm_path = paths[0] / S0
        
        # Pad/truncate to match training length (31 steps = n+1)
        target_length = 31
        if len(norm_path) < target_length:
            norm_path = np.pad(norm_path, (0, target_length - len(norm_path)), 
                              constant_values=norm_path[-1])
        else:
            norm_path = norm_path[:target_length]
        
        # Create parameter vector
        params = np.array([
            K / S0,
            B / S0,
            T / 2.0,
            r * 10,
            sigma * 2,
            1.0 if barrier_short == 'up' else 0.0
        ])
        
        # Predict
        path_input = norm_path.reshape(1, -1, 1)
        param_input = params.reshape(1, -1)
        rnn_price = model.predict([path_input, param_input], verbose=0)[0][0]
        rnn_price = max(rnn_price, 0.0)  # No negative prices
        
        # Calculate error
        if binomial_price > 0.01:
            rnn_error = abs(rnn_price - binomial_price) / binomial_price * 100
        else:
            rnn_error = 0

# ================= SIMULATE PATHS FOR VISUALIZATION =================
paths, knocked_out = simulate_paths(S0, B, T, r, sigma, n, num_paths, barrier_short)
knockout_prob = np.mean(knocked_out)

# ================= DASHBOARD =================
st.header("💰 Pricing Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📐 Binomial Tree Price", f"${binomial_price:.4f}", 
              help="Exact CRR binomial lattice with barrier monitoring")

with col2:
    if rnn_price:
        delta = rnn_price - binomial_price
        st.metric("🤖 RNN-LSTM Price", f"${rnn_price:.4f}", 
                 f"{delta:+.4f}" if abs(delta) > 0.0001 else "±0.0000")
    else:
        st.metric("🤖 RNN-LSTM Price", "Not trained")

with col3:
    if rnn_error:
        if rnn_error < 5:
            st.metric("📊 Prediction Error", f"{rnn_error:.2f}%", delta="Excellent", delta_color="normal")
        elif rnn_error < 15:
            st.metric("📊 Prediction Error", f"{rnn_error:.2f}%", delta="Good", delta_color="off")
        else:
            st.metric("📊 Prediction Error", f"{rnn_error:.2f}%", delta="High", delta_color="inverse")
    else:
        st.metric("📊 Prediction Error", "N/A")

with col4:
    st.metric("🎲 Knockout Probability", f"{knockout_prob:.1%}")

# ================= VISUALIZATION =================
st.header("📉 Simulated Stock Price Paths")

fig, ax = plt.subplots(figsize=(12, 6))
time_steps = np.linspace(0, T, n + 1)

for i in range(min(num_paths, len(paths))):
    color = 'red' if knocked_out[i] else 'green'
    alpha = 0.4 if knocked_out[i] else 0.7
    ax.plot(time_steps, paths[i], color=color, alpha=alpha, linewidth=1.0)

ax.axhline(B, color='black', linestyle='--', linewidth=2, label=f'Barrier (B={B:.1f})')
ax.axhline(K, color='orange', linestyle=':', linewidth=2, label=f'Strike (K={K:.1f})')
ax.axhline(S0, color='blue', linestyle='-.', alpha=0.5, label=f'Initial Price (S₀={S0:.1f})')

ax.set_xlabel("Time (Years)", fontsize=12)
ax.set_ylabel("Stock Price ($)", fontsize=12)
ax.set_title(f"{barrier_type.capitalize()} Call Option: {num_paths} Simulated Paths", fontsize=14)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

st.pyplot(fig)

st.caption("🔴 **Red paths:** Option knocked out (barrier breached) | 🟢 **Green paths:** Option survived")

# ================= EXPLANATION SECTION =================
with st.expander("📖 **How the RNN-LSTM Learns Option Pricing**", expanded=False):
    st.markdown("""
    ### 🎯 The Learning Task
    """)
    The RNN-LSTM model learns to approximate the binomial tree pricing function:
    
