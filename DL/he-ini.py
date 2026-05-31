import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
def activation(z):
    return z

def mlp_init(layer_sizes):
    weights_biases = []
    np.random.seed(42)
    for i in range(len(layer_sizes) - 1):
        std = np.sqrt(2 / layer_sizes[i])
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * std
        b = np.zeros((1, layer_sizes[i+1]))
        weights_biases.append((w, b))
    return weights_biases

def forward_layer(X, weights, biases):
    z = np.dot(X, weights) + biases
    a = activation(z)
    return a

def forward_pass(X, weights_biases):
    a = X
    for i, (w, b) in enumerate(weights_biases):
        a = forward_layer(a, w, b)
        print(f"Layer {i+1} output shape: {a.shape}")
    
    return a

def calculate_error(y_true, y_pred):
    error = y_true - y_pred
    mse = np.mean(error ** 2)
    return mse, error

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)
    print(f"\nDataset: Iris")
    print(f"Samples: {X.shape[0]}")
    print(f"Input features: {X.shape[1]}")
    print(f"Output classes: {y_onehot.shape[1]}")
    print(f"X shape: {X_scaled.shape}")
    print(f"y shape: {y_onehot.shape}") 
    layer_sizes = [4, 3]  
    print(f"\nNetwork architecture: {layer_sizes}")
    print("\n" + "="*60)
    print("= INITIALIZING MLP WITH RANDOM WEIGHTS "+ " "*20 + "=")
    print("="*60)
    weights_biases = mlp_init(layer_sizes)
    for i, (w, b) in enumerate(weights_biases):
        print(f"\nLayer {i+1} ({layer_sizes[i]} → {layer_sizes[i+1]}):")
        print(f"  Weights shape: {w.shape}")
        print(f"  Weights (first 2 rows):\n  {w[:2]}")
        print(f"  Biases: {b}")

    y_pred = forward_pass(X_scaled, weights_biases)
    mse, error = calculate_error(y_onehot, y_pred)
    print("\n" + "="*60)
    print("= ERROR CALCULATION"+" "*40 + "=")
    print("="*60)
    print(f"\nMean Squared Error (MSE): {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mse):.6f}")  
    print(f"\nError statistics:")
    print(f"  Min error: {error.min():.4f}")
    print(f"  Max error: {error.max():.4f}")
    print(f"  Mean error: {error.mean():.4f}")
    print(f"  Std error: {error.std():.4f}")
    print("\n" + "="*60)
    print("= SAMPLE RESULTS (FIRST 5 SAMPLES)"+" "*25 + "=")
    print("="*60)
    print(f"{'Sample':<6} | {'True':<20} | {'Predicted':<20} | {'Error':<20}")
    print("-"*75)
    
    for i in range(5):
        true_str = np.array2string(y_onehot[i], precision=2, suppress_small=True)
        pred_str = np.array2string(y_pred[i], precision=2, suppress_small=True)
        err_str = np.array2string(error[i], precision=2, suppress_small=True)
        print(f"{i:<6} | {true_str:<20} | {pred_str:<20} | {err_str:<20}")
    
    print("\n" + "="*60)
    print("= SUMMARY"+ " "*50 + "=")
    print("="*60)
    print(f"Total samples processed: {X.shape[0]}")
    print(f"Final MSE: {mse:.6f}")
    print(f"Final RMSE: {np.sqrt(mse):.6f}")
    print("="*60)

if __name__ == "__main__":
    main()