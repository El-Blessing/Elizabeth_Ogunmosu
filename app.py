import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pricer import price_barrier_option
from simulator import simulate_paths

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
# SIMULATOR FUNCTION (Fewer, Cleaner Paths)
# ==========================================
def simulate_paths(S0, B, T, r, sigma, n, num_paths=8, barrier_type='up'):
    """
    Generates risk-neutral GBM paths for visualization - MINIMAL VERSION.
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
# STREAMLIT DASHBOARD - CLEAN VERSION
# ==========================================
st.set_page_config(page_title="Barrier Option Visualizer", layout="wide")
st.title("📊 Barrier Option Path Visualizer")
st.markdown("See how barrier options work - **fewer paths, clearer picture**")

st.sidebar.header("Contract Parameters")

S0 = st.sidebar.slider("Initial Price (S₀)", 50.0, 150.0, 100.0, 1.0)
K = st.sidebar.slider("Strike (K)", 50.0, 150.0, 100.0, 1.0)
B = st.sidebar.slider("Barrier (B)", 50.0, 150.0, 120.0, 1.0)
T = st.sidebar.slider("Maturity (T, years)", 0.1, 2.0, 1.0, 0.1)
r = st.sidebar.slider("Risk-free Rate (r)", 0.0, 0.1, 0.05, 0.01)
sigma = st.sidebar.slider("Volatility (σ)", 0.05, 0.8, 0.3, 0.05)
n = st.sidebar.slider("Time Steps (n)", 20, 200, 50, 10)
barrier_type = st.sidebar.selectbox("Barrier Type", ["up", "down"])
num_paths = st.sidebar.slider("Number of Paths", 3, 12, 6, 1)  # REDUCED: max 12, default 6

# Auto-correct barrier constraints
if barrier_type == "up" and B <= S0:
    B = S0 + 5
    st.sidebar.warning(f"Barrier adjusted to {B} (must be above S₀)")
elif barrier_type == "down" and B >= S0:
    B = S0 - 5
    st.sidebar.warning(f"Barrier adjusted to {B} (must be below S₀)")

# Set random seed for reproducibility
np.random.seed(42)

# Compute price & simulate paths
price = price_barrier_option(S0, K, B, T, r, sigma, n, barrier_type)
paths, knocked_out = simulate_paths(S0, B, T, r, sigma, n, num_paths, barrier_type)
ko_prob = np.mean(knocked_out)

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("💰 Binomial Price", f"${price:.4f}")
with col2:
    st.metric("🎯 Knockout Probability", f"{ko_prob:.0%}")
with col3:
    st.metric("📈 Volatility", f"{sigma:.1%}")

# ==========================================
# CLEAN PLOT - THINNER LINES, LESS CLUTTER
# ==========================================
fig, ax = plt.subplots(figsize=(12, 6))

time_steps = np.linspace(0, T, n + 1)

# Plot paths with thinner lines and transparency
for i in range(num_paths):
    if knocked_out[i]:
        color = '#ff6b6b'  # Soft red
        alpha = 0.7
        linewidth = 1.5
        label = 'Knocked out' if i == 0 else ''
    else:
        color = '#51cf66'  # Soft green
        alpha = 0.9
        linewidth = 2.0
        label = 'Survived' if i == 0 else ''
    
    ax.plot(time_steps, paths[i], color=color, alpha=alpha, linewidth=linewidth, label=label)

# Add barrier and strike lines (thicker, more visible)
ax.axhline(B, color='#d6336c', linestyle='--', linewidth=2.5, label=f'Barrier (B = {B:.1f})')
ax.axhline(K, color='#fd7e14', linestyle=':', linewidth=2.5, label=f'Strike (K = {K:.1f})')
ax.axhline(S0, color='#adb5bd', linestyle='-', linewidth=1, alpha=0.5, label=f'Initial (S₀ = {S0:.1f})')

# Styling
ax.set_xlabel("Time (Years)", fontsize=12)
ax.set_ylabel("Stock Price ($)", fontsize=12)
ax.set_title(f"📊 {num_paths} Simulated Paths - {barrier_type}-and-out Call Option", fontsize=14, fontweight='bold')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.2, linestyle='--')
ax.set_facecolor('#f8f9fa')

st.pyplot(fig)

# ==========================================
# SIMPLE EXPLANATION
# ==========================================
st.info(f"""
**💡 What you're seeing:**
- 🟢 **Green paths** → Never hit the barrier → Payoff = max(Stock Price - ${K:.0f}, 0)
- 🔴 **Red paths** → Hit the barrier at some point → Payoff = $0 (knocked out)
- 📊 **Current price:** ${price:.4f} (calculated using a binomial tree with {n} time steps)
""")

# Optional: Show which paths knocked out
with st.expander("🔍 See which paths knocked out"):
    for i in range(num_paths):
        status = "🔴 KNOCKED OUT" if knocked_out[i] else "🟢 SURVIVED"
        final_price = paths[i, -1]
        payoff = max(final_price - K, 0) if not knocked_out[i] else 0
        st.write(f"Path {i+1}: {status} | Final: ${final_price:.2f} | Payoff: ${payoff:.2f}")
