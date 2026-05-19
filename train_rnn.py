"""
Train ML Model for Barrier Option Pricing
Uses Random Forest instead of TensorFlow for faster deployment
"""
import numpy as np
from pricer import price_barrier_option
from simulator import simulate_paths
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

def generate_training_data(n_samples=3000):
    """Generate training data using binomial tree"""
    print(f"Generating {n_samples} training samples...")
    X = []  # Features
    y = []  # Prices
    
    for i in range(n_samples):
        # Random parameters
        S0 = np.random.uniform(80, 120)
        K = np.random.uniform(70, 130)
        
        # Random barrier
        if np.random.rand() > 0.5:
            B = np.random.uniform(S0 + 5, min(150, S0 + 35))
            barrier_type = 'up'
        else:
            B = np.random.uniform(max(50, S0 - 35), S0 - 5)
            barrier_type = 'down'
        
        T = np.random.uniform(0.25, 1.5)
        r = np.random.uniform(0.02, 0.07)
        sigma = np.random.uniform(0.20, 0.45)
        
        # Calculate true price
        true_price = price_barrier_option(S0, K, B, T, r, sigma, 40, barrier_type)
        
        # Only use valid prices
        if 0.10 < true_price < 40:
            # Features (no path simulation needed for training)
            features = [
                S0 / 100,           # Normalized spot
                K / 100,            # Normalized strike  
                B / 100,            # Normalized barrier
                T,                  # Time to expiry
                r * 100,            # Rate percentage
                sigma * 100,        # Volatility percentage
                1 if barrier_type == 'up' else 0  # Barrier direction
            ]
            X.append(features)
            y.append(true_price)
        
        if (i + 1) % 500 == 0:
            print(f"  Progress: {i+1}/{n_samples}")
    
    return np.array(X), np.array(y)

def main():
    print("=" * 50)
    print("Training Barrier Option Pricing Model")
    print("=" * 50)
    
    # Generate data
    X, y = generate_training_data(3000)
    print(f"\n✅ Generated {len(X)} training samples")
    print(f"Price range: ${y.min():.2f} to ${y.max():.2f}")
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    print("\n🚀 Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val_scaled)
    errors = np.abs(y_pred - y_val) / y_val * 100
    
    print(f"\n📊 Results on validation set:")
    print(f"  Mean Absolute Error: ${np.mean(np.abs(y_pred - y_val)):.4f}")
    print(f"  Mean Error Percentage: {np.mean(errors):.2f}%")
    print(f"  Median Error Percentage: {np.median(errors):.2f}%")
    print(f"  R² Score: {model.score(X_val_scaled, y_val):.4f}")
    
    # Save model
    joblib.dump(model, 'option_pricing_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    
    print("\n💾 Model saved successfully!")
    print("   Files: option_pricing_model.joblib, feature_scaler.joblib")

if __name__ == "__main__":
    main()
