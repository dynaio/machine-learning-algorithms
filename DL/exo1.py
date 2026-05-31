import math as m


n = 0.1  # learning rate

w1 = 0.5   
w2 = 0.1   
w3 = 0.62  
w4 = 0.2   
w5 = -0.2  
w6 = 0.3   
b1 = 0.4   
b2 = -0.1  
b3 = 1.83  

data = [
    (0.1, 0.3, 0.03)
]


def activation(z):
    return 1 / (1 + m.exp(-z))


def predict(x1, x2):
    # Hidden layer
    z1 = x1 * w1 + x2 * w2 + b1
    h1 = activation(z1)
    
    z2 = x1 * w3 + x2 * w4 + b2
    h2 = activation(z2)
    
    # Output layer
    z_out = h1 * w5 + h2 * w6 + b3
    y_hat = activation(z_out)
    
    return y_hat


def update():
    global w1, w2, w3, w4, w5, w6, b1, b2, b3
    
    x1, x2, target = data[0]
    y_hat = predict(x1, x2)
    
    # Error
    error = target - y_hat
    
    
    # Output layer
    d_y = error * y_hat * (1 - y_hat)   
    
    d_w5 = d_y * activation(x1 * w1 + x2 * w2 + b1)   # h1
    d_w6 = d_y * activation(x1 * w3 + x2 * w4 + b2)   # h2
    d_b3 = d_y
    
    # Hidden layer
    d_h1 = d_y * w5
    d_h2 = d_y * w6
    
    d_z1 = d_h1 * activation(x1 * w1 + x2 * w2 + b1) * (1 - activation(x1 * w1 + x2 * w2 + b1))
    d_z2 = d_h2 * activation(x1 * w3 + x2 * w4 + b2) * (1 - activation(x1 * w3 + x2 * w4 + b2))
    
    d_w1 = d_z1 * x1
    d_w2 = d_z1 * x2
    d_b1 = d_z1
    
    d_w3 = d_z2 * x1
    d_w4 = d_z2 * x2
    d_b2 = d_z2
    
    # ── Update ──────────────────────────────────────────
    w1 += n * d_w1
    w2 += n * d_w2
    w3 += n * d_w3
    w4 += n * d_w4
    w5 += n * d_w5
    w6 += n * d_w6
    b1 += n * d_b1
    b2 += n * d_b2
    b3 += n * d_b3
    
    return (target - y_hat) ** 2   


def train(epochs=300):
    print("Initial weights & biases:")
    print(f"w1={w1:.3f} w2={w2:.3f} w3={w3:.3f} w4={w4:.3f}")
    print(f"w5={w5:.3f} w6={w6:.3f} b1={b1:.3f} b2={b2:.3f} b3={b3:.3f}")
    y_init = predict(0.1, 0.3)
    print(f"\nInitial prediction: {y_init:.6f}  (target = 0.03)\n")
    
    for epoch in range(1, epochs + 1):
        sq_error = update()
        
        if epoch % 30 == 0 or epoch == 1 or epoch == epochs:
            y = predict(0.1, 0.3)
            print(f"Epoch {epoch:4d} | sq_error = {sq_error:.6f} | y = {y:.6f}")
    
    print("\nFinal weights & biases:")
    print(f"w1={w1:.4f} w2={w2:.4f} w3={w3:.4f} w4={w4:.4f}")
    print(f"w5={w5:.4f} w6={w6:.4f} b1={b1:.4f} b2={b2:.4f} b3={b3:.4f}")
    
    y_final = predict(0.1, 0.3)
    print(f"\nFinal prediction: {y_final:.6f}  (target = 0.03)")

# Run training
train(300)

# B. Abdelkader – 2026 😊