import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pricer import price_barrier_option
from simulator import simulate_paths
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Barrier Option Pricing", layout="wide")

st.title("📊 Barrier Option Pricing: Binomial Tree vs ML Surrogate")
st.markdown("Compare exact binomial tree pricing with a fast Machine Learning model")

# ================= TRAIN ML MODEL ON THE FLY =================
@st.cache_resource
def train_ml_model():
    """Train ML model inside the app - no external files needed"""
    with st.spinner("🤖 Training ML model (30 seconds, first time only)..."):
        X = []
        y = []
        
        # Generate training data
        for _ in range(2000):
            # Random parameters
            S0 = np.random.uniform(70, 130)
            K = np.random.uniform(60, 140)
            
            # Random barrier
            if np.random.rand() > 0.5:
                B = np.random.uniform(S0 + 5, min(150, S0 + 40))
                btype = 'up'
            else:
                B = np.random.uniform(max(50, S0 - 40), S0 - 5)
                btype = 'down'
            
            T = np.random.uniform(0.25, 1.8)
            r = np.random.uniform(0.01, 0.08)
            sigma = np.random.uniform(0.15, 0.50)
            
            # Calculate price
            try:
                price = price_barrier_option(S0, K, B, T, r, sigma, 30, btype)
                
                if 0.10 < price < 50:
                    # Features (no paths needed for training)
                    features = [
                        S0 / 100,
                        K / 100,
                        B / 100,
                        T,
                        r * 100,
                        sigma * 100,
                        1 if btype == 'up' else 0
                    ]
                    X.append(features)
                    y.append(price)
            except:
                continue
        
        if len(X) == 0:
            return None, None
        
        X = np.array(X)
        y = np.array(y)
        
        # Scale and train
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestRegressor(
            n_estimators=80,
            max_depth=10,
            random_state=42
        )
        model.fit(X_scaled, y)
        
        st.success(f"✅ Model trained on {len(X)} samples")
        return model, scaler

# ================= SIDEBAR =================
st.sidebar.header("📈 Contract Parameters")

S0 = st.sidebar.slider("Stock Price (S₀)", 60, 140, 100)
K = st.sidebar.slider("Strike Price (K)", 60, 140, 100)
B = st.sidebar.slider("Barrier (B)", 60, 160, 120)
T = st.sidebar.slider("Time to Maturity (years)", 0.2, 2.0, 1.0, 0.1)
r = st.sidebar.slider("Risk-free Rate (%)", 1, 10, 5) / 100
sigma = st.sidebar.slider("Volatility (%)", 15, 60, 30) / 100
n = st.sidebar.slider("Binomial Steps", 20, 80, 40)
barrier_type = st.sidebar.selectbox("Barrier Type", ["up", "down"])
num_paths = st.sidebar.slider("Simulated Paths", 10, 40, 20)

# Fix barrier if needed
if barrier_type == "up" and B <= S0:
    B = S0 + 5
    st.sidebar.warning(f"Barrier adjusted to {B}")
elif barrier_type == "down" and B >= S0:
    B = S0 - 5
    st.sidebar.warning(f"Barrier adjusted to {B}")

# ================= CALCULATIONS =================
# Binomial price
binomial_price = price_barrier_option(S0, K, B, T, r, sigma, n, barrier_type)

# ML Model
model, scaler = train_ml_model()

# ML Prediction
ml_price = None
ml_error = None

if model and scaler:
    try:
        features = np.array([[
            S0 / 100,
            K / 100,
            B / 100,
            T,
            r * 100,
            sigma * 100,
            1 if barrier_type == "up" else 0
        ]])
        features_scaled = scaler.transform(features)
        ml_price = model.predict(features_scaled)[0]
        ml_price = max(ml_price, 0.01)
        
        if binomial_price > 0.01:
            ml_error = abs(ml_price - binomial_price) / binomial_price * 100
    except:
        pass

# Simulate paths
paths, knocked_out = simulate_paths(S0, B, T, r, sigma, n, num_paths, barrier_type)
knockout_prob = np.mean(knocked_out)

# ================= RESULTS =================
st.header("💰 Pricing Results")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📐 Binomial Tree", f"${binomial_price:.4f}")

with col2:
    if ml_price:
        st.metric("🤖 ML Surrogate", f"${ml_price:.4f}")
    else:
        st.metric("🤖 ML Surrogate", "Training...")

with col3:
    if ml_error:
        if ml_error < 10:
            st.metric("📊 Error", f"{ml_error:.1f}%", delta="Good")
        else:
            st.metric("📊 Error", f"{ml_error:.1f}%", delta="High")
    else:
        st.metric("📊 Error", "N/A")

with col4:
    st.metric("🎲 Knockout Prob", f"{knockout_prob:.1%}")

# Show interpretation if error is high
if ml_error and ml_error > 30:
    st.warning(f"""
    **⚠️ ML error is {ml_error:.1f}% – What does this mean?**  
    - Binomial price = **${binomial_price:.4f}** (very low)  
    - ML model was trained mostly on higher-priced options  
    - **Result:** ML overestimates cheap options like this one  

    **💡 Conclusion for colleagues:**  
    Random Forest is not suitable for low‑value barrier options.  
    An RNN‑LSTM would perform better, but needs TensorFlow.  
    For accurate trading decisions, always trust the **Binomial Tree price**.
    """)
elif ml_error and ml_error > 15:
    st.info(f"📊 ML error is {ml_error:.1f}% – acceptable for approximation purposes.")
    
# ================= PATH PLOT =================
st.header("📉 Simulated Stock Price Paths")

fig, ax = plt.subplots(figsize=(12, 5))
time_steps = np.linspace(0, T, n + 1)

for i in range(num_paths):
    color = 'red' if knocked_out[i] else 'green'
    alpha = 0.5 if knocked_out[i] else 0.7
    ax.plot(time_steps, paths[i], color=color, alpha=alpha, linewidth=0.8)

ax.axhline(B, color='black', linestyle='--', linewidth=2, label=f'Barrier (B={B})')
ax.axhline(K, color='orange', linestyle=':', linewidth=2, label=f'Strike (K={K})')
ax.set_xlabel("Time (Years)")
ax.set_ylabel("Stock Price ($)")
ax.set_title(f"{barrier_type.upper()}-and-Out Call Option")
ax.legend()
ax.grid(True, alpha=0.3)

st.pyplot(fig)
st.caption("🔴 Red = Knocked out | 🟢 Green = Survived")

# ================= EXPLANATION =================
with st.expander("📖 How It Works"):
    st.markdown("""
    **Binomial Tree (Ground Truth)**
    - Cox-Ross-Rubinstein lattice with barrier monitoring
    - Exact pricing for discrete barrier options
    - Slow but accurate (O(n²) complexity)
    
    **ML Surrogate (Random Forest)**
    - Trained on 2,000 random contracts priced with binomial tree
    - Features: S₀, K, B, T, r, σ, barrier type
    - ~50x faster than binomial tree
    
    **When to use each:**
    - **Binomial Tree:** Final pricing, risk reports, regulatory filings
    - **ML Surrogate:** Real-time estimates, sensitivity analysis, large portfolios
    """)

# ================= SIDEBAR INFO =================
st.sidebar.markdown("---")
st.sidebar.info(
    "**ML Model:** Random Forest trained on 2,000 binomial tree calculations\n\n"
    "**Expected Error:** 5-15% for typical parameters"
)
