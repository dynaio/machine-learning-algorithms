import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

# ==================== ACTIVATION FUNCTIONS ====================
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # Clip to avoid overflow

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# ==================== LOSS FUNCTIONS ====================
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

# ==================== INITIALIZATION FUNCTIONS ====================

#    Xavier/Glorot initialization 
def initialize_weights(layer_sizes):

    weights_biases = []
    np.random.seed(42)
    
    for i in range(len(layer_sizes) - 1):
        # Xavier initialization: variance = 2/(n_in + n_out)
        limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
        w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
        b = np.zeros((1, layer_sizes[i+1]))
        weights_biases.append((w, b))
    
    return weights_biases

#     He initialization
def initialize_weights(layer_sizes):

    weights_biases = []
    np.random.seed(42)
    
    for i in range(len(layer_sizes) - 1):
        # He initialization: variance = 2/n_in
        w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
        b = np.zeros((1, layer_sizes[i+1]))
        weights_biases.append((w, b))
    
    return weights_biases

#    Random initialization 
def initialize_weights(layer_sizes):

    weights_biases = []
    np.random.seed(42)
    
    for i in range(len(layer_sizes) - 1):
        w = np.random.rand(layer_sizes[i], layer_sizes[i+1]) - 0.5
        b = np.zeros((1, layer_sizes[i+1]))
        weights_biases.append((w, b))
    
    return weights_biases
#    Zeros initialization 
def initialize_weights(layer_sizes):
    weights_biases = []
    np.random.seed(42)
    
    for i in range(len(layer_sizes) - 1):
        w = np.zeros((layer_sizes[i], layer_sizes[i+1]))
        b = np.zeros((1, layer_sizes[i+1]))
        weights_biases.append((w, b))
    
    return weights_biases

# ==================== FORWARD PASS ====================
def forward_layer(input_data, weights, biases, activation='relu'):
    z = np.dot(input_data, weights) + biases
    
    if activation == 'relu':
        a = relu(z)
    elif activation == 'sigmoid':
        a = sigmoid(z)
    elif activation == 'softmax':
        a = softmax(z)
    else:  # linear
        a = z
    
    return a, (input_data, z, weights, activation)

def forward_pass(X, weights_biases):
    caches = []
    a = X
    # Hidden layers (ReLU)
    for i in range(len(weights_biases) - 1):
        w, b = weights_biases[i]
        a, cache = forward_layer(a, w, b, activation='relu')
        caches.append(cache)
    # Output layer (softmax)
    w, b = weights_biases[-1]
    y_pred, cache = forward_layer(a, w, b, activation='softmax')
    caches.append(cache)
    return y_pred, caches
# ==================== BACKWARD PASS ====================
def backward_layer(dA, cache, y_true=None):

    A_prev, Z, W, activation = cache
    m = A_prev.shape[0]
    if activation == 'softmax' and y_true is not None:
        dZ = dA  
    elif activation == 'relu':
        dZ = dA * relu_derivative(Z)
    else:
        dZ = dA
    dW = np.dot(A_prev.T, dZ) / m
    db = np.sum(dZ, axis=0, keepdims=True) / m
    dA_prev = np.dot(dZ, W.T)   
    return dA_prev, dW, db

def backward_pass(y_true, y_pred, caches):
    gradients = []
    m = y_true.shape[0]
    dA = y_pred - y_true  
    for i in range(len(caches) - 1, -1, -1):
        if i == len(caches) - 1: 
            dA_prev, dW, db = backward_layer(dA, caches[i], y_true)
        else:  # Hidden layers
            dA_prev, dW, db = backward_layer(dA, caches[i])
        
        gradients.insert(0, (dW, db))
        dA = dA_prev
    
    return gradients

# ==================== UPDATE PARAMETERS ====================
def update_parameters(weights_biases, gradients, learning_rate):

    new_weights_biases = []
    
    for i, ((w, b), (dw, db)) in enumerate(zip(weights_biases, gradients)):
        w_new = w - learning_rate * dw
        b_new = b - learning_rate * db
        new_weights_biases.append((w_new, b_new))
    
    return new_weights_biases

# ==================== TRAINING FUNCTION ====================
def train_mlp(X_train, y_train, layer_sizes, epochs=10, learning_rate=0.01):

    weights_biases = initialize_weights(layer_sizes)
    losses = []
    print("="*80)
    print("MLP TRAINING - INITIAL WEIGHTS AND BIASES")
    print("="*80)
    for i, (w, b) in enumerate(weights_biases):
        print(f"\nLayer {i+1} - Shape: {w.shape}")
        print(f"Weights:\n{w}")
        print(f"Biases:\n{b}")
    
    for epoch in range(epochs):
        y_pred, caches = forward_pass(X_train, weights_biases)
        loss = cross_entropy_loss(y_train, y_pred)
        losses.append(loss)        
        gradients = backward_pass(y_train, y_pred, caches)
        
        old_weights_biases = weights_biases.copy()
        weights_biases = update_parameters(weights_biases, gradients, learning_rate)
        
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}")
        print(f"Loss: {loss:.6f}")
        print(f"{'='*80}")
        
        for i, ((w_old, b_old), (w_new, b_new), (dw, db)) in enumerate(zip(old_weights_biases, weights_biases, gradients)):
            print(f"\nLayer {i+1}:")
            print(f"Weight changes:")
            print(f"  Old: {w_old[0, :3]}..." if w_old.shape[1] > 3 else f"  Old: {w_old}")
            print(f"  Grad: {dw[0, :3]}..." if dw.shape[1] > 3 else f"  Grad: {dw}")
            print(f"  New: {w_new[0, :3]}..." if w_new.shape[1] > 3 else f"  New: {w_new}")
            print(f"  Update magnitude: {np.linalg.norm(dw):.6f}")
            
            print(f"Bias changes:")
            print(f"  Old: {b_old}")
            print(f"  Grad: {db}")
            print(f"  New: {b_new}")
        
        sample_pred = y_pred[:3]
        sample_true = y_train[:3] 
        print(f"\nSample predictions (first 3 samples):")
        for i in range(3):
            pred_class = np.argmax(sample_pred[i])
            true_class = np.argmax(sample_true[i])
            print(f"  Sample {i+1}: Pred={pred_class} (prob={sample_pred[i][pred_class]:.4f}), True={true_class}")
    
    return weights_biases, losses

# ==================== MAIN EXECUTION ====================
def load_and_prepare_iris():
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y)
    
    return X_scaled, y_onehot

def main():
    # Load data
    n_epochs = 2000
    X, y = load_and_prepare_iris()
    print("Dataset shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"Classes: {y.shape[1]}")
    

    layer_sizes = [4, 3, 2, 3]
    
    trained_weights, losses = train_mlp(
        X_train=X,
        y_train=y,
        layer_sizes=layer_sizes,
        epochs=n_epochs,
        learning_rate=0.1
    )
    
    y_pred, _ = forward_pass(X, trained_weights)
    accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Final Training Accuracy: {accuracy*100:.2f}%")
    print(f"Final Loss: {losses[-1]:.6f}")
    
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, n_epochs+1), losses, 'bo-')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("\nLoss progression:")
        for i, loss in enumerate(losses, 1):
            print(f"Epoch {i}: {loss:.6f}")

if __name__ == "__main__":
    main()