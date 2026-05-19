"""
RNN-LSTM Training Script for Barrier Option Pricing
Trains a neural network to learn the binomial tree pricing function
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pricer import price_barrier_option
from simulator import simulate_paths
import pickle

def generate_training_data(n_samples=5000, max_steps=30):
    """
    Generate training data using binomial tree as ground truth
    Each sample: features (path + contract params) -> price
    """
    print(f"Generating {n_samples} training samples...")
    X_paths = []      # Input: normalized price paths
    X_params = []     # Input: contract parameters
    y_prices = []     # Output: option prices
    
    for i in range(n_samples):
        # Random contract parameters (realistic ranges)
        S0 = np.random.uniform(80, 120)
        K = np.random.uniform(70, 130)
        
        # Ensure barrier is realistic
        if np.random.rand() > 0.5:
            B = np.random.uniform(S0 + 5, min(150, S0 + 40))
            barrier_type = 'up'
        else:
            B = np.random.uniform(max(50, S0 - 40), S0 - 5)
            barrier_type = 'down'
        
        T = np.random.uniform(0.25, 2.0)
        r = np.random.uniform(0.01, 0.08)
        sigma = np.random.uniform(0.15, 0.50)
        n_steps = max_steps
        
        # Get ground truth price from binomial tree
        true_price = price_barrier_option(S0, K, B, T, r, sigma, n_steps, barrier_type)
        
        # Only use samples with meaningful prices (not zero or too small)
        if true_price < 0.05 or true_price > 50:
            continue
            
        # Generate a single price path
        paths, knocked = simulate_paths(S0, B, T, r, sigma, n_steps, 1, barrier_type)
        
        # Normalize path by initial price
        norm_path = paths[0] / S0
        
        # Pad or truncate to fixed length
        if len(norm_path) < max_steps + 1:
            norm_path = np.pad(norm_path, (0, max_steps + 1 - len(norm_path)), 
                              constant_values=norm_path[-1])
        else:
            norm_path = norm_path[:max_steps + 1]
        
        # Create parameter vector (normalized)
        params = np.array([
            K / S0,           # Strike relative to spot
            B / S0,           # Barrier relative to spot  
            T / 2.0,          # Time to expiry (normalized)
            r * 10,           # Rate scaled
            sigma * 2,        # Volatility scaled
            1.0 if barrier_type == 'up' else 0.0  # Barrier type
        ])
        
        X_paths.append(norm_path)
        X_params.append(params)
        y_prices.append(true_price)
        
        if (i + 1) % 1000 == 0:
            print(f"Generated {i + 1}/{n_samples} samples...")
    
    X_paths = np.array(X_paths)
    X_params = np.array(X_params)
    y_prices = np.array(y_prices)
    
    print(f"\n✅ Generated {len(X_paths)} valid samples")
    print(f"Price range: ${y_prices.min():.2f} - ${y_prices.max():.2f}")
    print(f"Mean price: ${y_prices.mean():.2f}")
    
    return X_paths, X_params, y_prices

def build_rnn_lstm_model(path_length, n_params):
    """
    Build a hybrid RNN-LSTM model that takes both path and parameters
    """
    # Path input (the price sequence)
    path_input = layers.Input(shape=(path_length, 1), name='price_path')
    
    # LSTM layers to process the sequence
    x = layers.LSTM(64, return_sequences=True)(path_input)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    path_features = layers.Dense(16, activation='relu')(x)
    
    # Parameter input (contract features)
    param_input = layers.Input(shape=(n_params,), name='contract_params')
    y = layers.Dense(16, activation='relu')(param_input)
    y = layers.Dense(8, activation='relu')(y)
    param_features = y
    
    # Combine both paths
    combined = layers.Concatenate()([path_features, param_features])
    z = layers.Dense(32, activation='relu')(combined)
    z = layers.Dense(16, activation='relu')(z)
    output = layers.Dense(1, name='option_price')(z)
    
    # Create model
    model = keras.Model(inputs=[path_input, param_input], outputs=output)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',  # More robust to outliers than MSE
        metrics=['mae', 'mape']
    )
    
    return model

def train_and_save_model():
    """
    Main training function
    """
    print("=" * 60)
    print("RNN-LSTM Training for Barrier Option Pricing")
    print("=" * 60)
    
    # Generate data
    X_paths, X_params, y_prices = generate_training_data(n_samples=8000, max_steps=30)
    
    # Train/validation split
    split_idx = int(0.8 * len(X_paths))
    X_paths_train = X_paths[:split_idx]
    X_params_train = X_params[:split_idx]
    y_train = y_prices[:split_idx]
    
    X_paths_val = X_paths[split_idx:]
    X_params_val = X_params[split_idx:]
    y_val = y_prices[split_idx:]
    
    # Build model
    model = build_rnn_lstm_model(
        path_length=X_paths.shape[1],
        n_params=X_params.shape[1]
    )
    
    print("\n📊 Model Architecture:")
    model.summary()
    
    # Train with callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
    
    print("\n🚀 Training model...")
    history = model.fit(
        [X_paths_train, X_params_train], y_train,
        validation_data=([X_paths_val, X_params_val], y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n📈 Evaluation on validation set:")
    val_loss, val_mae, val_mape = model.evaluate([X_paths_val, X_params_val], y_val, verbose=0)
    print(f"Validation MAE: ${val_mae:.4f}")
    print(f"Validation MAPE: {val_mape:.2f}%")
    
    # Test on a few examples
    print("\n🔍 Sample predictions:")
    test_indices = np.random.choice(len(X_paths_val), 5, replace=False)
    for idx in test_indices:
        true_price = y_val[idx]
        pred_price = model.predict(
            [X_paths_val[idx:idx+1], X_params_val[idx:idx+1]], 
            verbose=0
        )[0][0]
        error_pct = abs(pred_price - true_price) / true_price * 100
        print(f"  True: ${true_price:.4f} → Pred: ${pred_price:.4f} | Error: {error_pct:.1f}%")
    
    # Save model and scalers
    model.save('barrier_rnn_model.h5')
    print("\n💾 Model saved as 'barrier_rnn_model.h5'")
    
    return model, history

if __name__ == "__main__":
    train_and_save_model()
