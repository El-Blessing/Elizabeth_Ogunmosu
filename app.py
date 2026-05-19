import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pricer import price_barrier_option
from simulator import simulate_paths

st.set_page_config(page_title="Barrier Option Visualizer", layout="wide")
st.title("📊 Interactive Barrier Option Path Visualizer & Pricer")
st.markdown("Demonstrates **path-dependence** and the computational motivation for RNN surrogates.")

st.sidebar.header("Contract Parameters")
S0 = st.sidebar.slider("Initial Price (S₀)", 50.0, 150.0, 100.0, 1.0)
K = st.sidebar.slider("Strike (K)", 50.0, 150.0, 100.0, 1.0)
B = st.sidebar.slider("Barrier (B)", 50.0, 150.0, 120.0, 1.0)
T = st.sidebar.slider("Maturity (T, years)", 0.1, 2.0, 1.0, 0.1)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.1, 0.05, 0.01)
sigma = st.sidebar.slider("Volatility (σ)", 0.05, 0.8, 0.3, 0.05)
n = st.sidebar.slider("Time Steps (n)", 20, 200, 50, 10)
barrier_type = st.sidebar.selectbox("Barrier Type", ["up", "down"])
num_paths = st.sidebar.slider("Simulated Paths", 10, 100, 30)

# Auto-correct barrier constraints for demo stability
if barrier_type == "up" and B <= S0:
    B = S0 + 5
elif barrier_type == "down" and B >= S0:
    B = S0 - 5

# Compute price & simulate paths
price = price_barrier_option(S0, K, B, T, r, sigma, n, barrier_type)
paths, knocked_out = simulate_paths(S0, B, T, r, sigma, n, num_paths, barrier_type)
ko_prob = np.mean(knocked_out)

col1, col2 = st.columns(2)
with col1:
    st.metric("Binomial Price (CRR)", f"${price:.4f}")
    st.metric("Knockout Probability (MC)", f"{ko_prob:.2%}")
with col2:
    st.info("💡 **Path Dependence:** Two paths ending at the same price yield different payoffs if one crosses the barrier. This is why RNNs process *sequences*, not just static parameters.")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
time_steps = np.linspace(0, T, n + 1)
for i in range(num_paths):
    color = 'red' if knocked_out[i] else 'green'
    alpha = 0.6 if knocked_out[i] else 0.9
    ax.plot(time_steps, paths[i], color=color, alpha=alpha, linewidth=1.5)
ax.axhline(B, color='black', linestyle='--', label=f'Barrier (B={B:.1f})')
ax.axhline(K, color='orange', linestyle=':', label=f'Strike (K={K:.1f})')
ax.set_xlabel("Time (Years)")
ax.set_ylabel("Stock Price")
ax.set_title(f"Simulated Risk-Neutral Paths ({barrier_type}-and-out Call)")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.caption("🔴 Red paths = knocked out | 🟢 Green paths = survived to maturity")
