import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the BMW dataset
data = pd.read_csv('DL/bmw_processed.csv')
print("Dataset head:")
print(data.head())
print(f"\nDataset shape: {data.shape}")

# Assuming the last column is the target (based on your output showing 'Target' column)
# Modify this based on your actual target column name
X = data.drop('Target', axis=1).values  # Features
y = data['Target'].values  # Target

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better neural network performance
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale the target as well (optional, but can help with regression)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# Create and train the neural network
model = MLPRegressor(
    hidden_layer_sizes=(2,),  # One hidden layer with 2 neurons
    activation='tanh',
    solver='sgd',
    learning_rate_init=0.1,
    max_iter=500,  # Increased iterations for real dataset
    random_state=42,
    verbose=True  # Show training progress
)

# Train the model
print("\nTraining the neural network...")
model.fit(X_train_scaled, y_train_scaled)

# Make predictions
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Calculate and print performance metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance on Test Set:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Show some sample predictions
print(f"\nSample Predictions (first 5 test samples):")
for i in range(min(5, len(y_test))):
    print(f"Predicted: {y_pred[i]:.4f}, Actual: {y_test[i]:.4f}")

# Print the weights and biases
print("\nWeights and biases:")
for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
    print(f"Layer {i+1} weights:\n{coef}")
    print(f"Layer {i+1} biases:\n{intercept}")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(model.loss_curve_)
plt.title("Training Error over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predictions vs Actual")
plt.grid(True)

plt.tight_layout()
plt.show()

feature_importance = np.abs(model.coefs_[0]).mean(axis=1)
feature_names = data.drop('Target', axis=1).columns
print("\nFeature Importance (based on average absolute weights in first layer):")
for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
    print(f"{name}: {importance:.4f}")