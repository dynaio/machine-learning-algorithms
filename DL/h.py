import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ────────────────────────────────────────────────
# Load and prepare data
# ────────────────────────────────────────────────
df = pd.read_csv('DL/bmw_processed.csv')

# Assuming Sales_Volume is the regression target
# Remove Target if it's a classification label you don't want to use here
if 'Target' in df.columns:
    df = df.drop(columns=['Target'])

# Choose features (all except the target)
target_col = 'Sales_Volume'
X = df.drop(columns=[target_col])
y = df[target_col].values

print("Features shape:", X.shape)
print("Target shape:", y.shape)
print("\nFirst few rows of features:\n", X.head())

# Scale features (very important for MLP!)
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Optional: scale target too (helps when values are large/small)
# scaler_y = StandardScaler()
# y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# ────────────────────────────────────────────────
# Train / test split
# ────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,           # or y_scaled if you scaled y
    test_size=0.2,
    random_state=42
)

# ────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────
model = MLPRegressor(
    hidden_layer_sizes=(8, 4),        # feel free to change — more capacity than (2,)
    activation='tanh',
    solver='sgd',
    learning_rate_init=0.05,          # usually smaller than 0.1 on real data
    momentum=0.9,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=50,
    random_state=42,
    verbose=False
)

model.fit(X_train, y_train)

# ────────────────────────────────────────────────
# Evaluate
# ────────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

train_mse = np.mean((y_pred_train - y_train)**2)
test_mse  = np.mean((y_pred_test  - y_test)**2)

print(f"\nFinal training   MSE: {train_mse:.6f}")
print(f"Final test       MSE: {test_mse:.6f}")
print(f"Final train RMSE: {np.sqrt(train_mse):.6f}")
print(f"Final test  RMSE: {np.sqrt(test_mse):.6f}")

# If you scaled y → inverse transform predictions
# y_pred_test_orig  = scaler_y.inverse_transform(y_pred_test.reshape(-1,1)).ravel()

# ────────────────────────────────────────────────
# Visualizations
# ────────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("True Sales Volume")
plt.ylabel("Predicted Sales Volume")
plt.title("Prediction vs True (test set)")
plt.grid(True)

plt.tight_layout()
plt.show()

# Optional: print final weights (only useful for very small nets)
if sum(s[0]*s[1] for s in model.coefs_) < 200:
    print("\nWeights and biases:")
    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_), 1):
        print(f"Layer {i} weights:\n{coef.round(5)}")
        print(f"Layer {i} biases:\n{intercept.round(5)}\n")