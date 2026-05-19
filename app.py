import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pricer import price_barrier_option
from simulator import simulate_paths

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# PRICER FUNCTION (CRR Binomial Tree)
# ==========================================
def price_barrier_option(S0, K, B, T, r, sigma, n, barrier_type='up'):
    """
    Cox-Ross-Rubinstein binomial pricer with discrete barrier monitoring.
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    p = np.clip(p, 0.0, 1.0)
    disc = np.exp(-r * dt)

    # Terminal stock prices & European call payoff
    j = np.arange(n + 1)
    S_T = S0 * (u ** j) * (d ** (n - j))
    V = np.maximum(S_T - K, 0.0)

    # Apply barrier condition at maturity
    if barrier_type == 'up':
        V[S_T >= B] = 0.0
    else:
        V[S_T <= B] = 0.0

    # Backward induction with discrete barrier check at each step
    for i in range(n - 1, -1, -1):
        j = np.arange(i + 1)
        S_i = S0 * (u ** j) * (d ** (i - j))
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])

        if barrier_type == 'up':
            V[S_i >= B] = 0.0
        else:
            V[S_i <= B] = 0.0

    return V[0]


# ==========================================
# SIMULATOR FUNCTION (GBM Paths)
# ==========================================
def simulate_paths(S0, B, T, r, sigma, n, num_paths=50, barrier_type='up'):
    """
    Generates risk-neutral GBM paths for visualization.
    """
    dt = T / n
    paths = np.zeros((num_paths, n + 1))
    paths[:, 0] = S0

    for t in range(1, n + 1):
        Z = np.random.randn(num_paths)
        paths[:, t] = paths[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Track barrier breaches
    if barrier_type == 'up':
        knocked_out = np.any(paths >= B, axis=1)
    else:
        knocked_out = np.any(paths <= B, axis=1)

    return paths, knocked_out


# ==========================================
# STREAMLIT DASHBOARD
# ==========================================
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
    st.sidebar.warning(f"Barrier adjusted to {B} (must be above S₀ for up-and-out)")
elif barrier_type == "down" and B >= S0:
    B = S0 - 5
    st.sidebar.warning(f"Barrier adjusted to {B} (must be below S₀ for down-and-out)")

# Compute price & simulate paths (using same 'n' consistently)
price = price_barrier_option(S0, K, B, T, r, sigma, n, barrier_type)
paths, knocked_out = simulate_paths(S0, B, T, r, sigma, n, num_paths, barrier_type)
ko_prob = np.mean(knocked_out)

# Display metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("Binomial Price (CRR)", f"${price:.4f}")
    st.metric("Knockout Probability (MC)", f"{ko_prob:.2%}")
with col2:
    st.info("💡 **Path Dependence:** Two paths ending at the same price yield different payoffs if one crosses the barrier. This is why RNNs process *sequences*, not just static parameters.")

# Plot paths
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

# Additional explanation
with st.expander("📖 How the Binomial Tree Works"):
    st.write(f"""
    - **Time steps:** {n} steps over {T} years → Δt = {T/n:.4f} years
    - **Up factor:** u = e^(σ√Δt) = {np.exp(sigma * np.sqrt(T/n)):.4f}
    - **Down factor:** d = 1/u = {1/np.exp(sigma * np.sqrt(T/n)):.4f}
    - **Risk-neutral probability:** p = {max(0, min(1, (np.exp(r * T/n) - 1/np.exp(sigma * np.sqrt(T/n))) / (np.exp(sigma * np.sqrt(T/n)) - 1/np.exp(sigma * np.sqrt(T/n))))):.4f}
    - **Binomial price:** ${price:.4f}
    """)
