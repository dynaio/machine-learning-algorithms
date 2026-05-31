import math as m

# Hyperparameters
n = 0.1

w11 = 0.1
w12 = 0.2
w13 = 0.3
w21 = 0.4
w22 = 0.5
w23 = 0.6
w31 = 0.1
w32 = 0.4
w41 = 0.2
w42 = 0.5
w51 = 0.3
w52 = 0.6

bh1 = 0.7
bh2 = 0.8
bh3 = 0.9

bo1 = 0.1
bo2 = 0.2

data = [
    (0.5, 0.8, 0.7, 0.4)
]

def sigmoid(z):
    return 1 / (1 + m.exp(-z))

def predict(x1, x2):
    z1 = x1 * w11 + x2 * w21 + bh1
    z2 = x1 * w12 + x2 * w22 + bh2
    z3 = x1 * w13 + x2 * w23 + bh3
    
    h1 = sigmoid(z1)
    h2 = sigmoid(z2)
    h3 = sigmoid(z3)
    
    zo1 = h1 * w31 + h2 * w41 + h3 * w51 + bo1
    zo2 = h1 * w32 + h2 * w42 + h3 * w52 + bo2
    
    y1_hat = sigmoid(zo1)
    y2_hat = sigmoid(zo2)
    
    return y1_hat, y2_hat

def update_batch():
    global w11, w12, w13, w21, w22, w23, w31, w32, w41, w42, w51, w52
    global bh1, bh2, bh3, bo1, bo2
    
    total_error = 0.0
    
    dw11 = dw12 = dw13 = dw21 = dw22 = dw23 = 0.0
    dw31 = dw32 = dw41 = dw42 = dw51 = dw52 = 0.0
    dbh1 = dbh2 = dbh3 = dbo1 = dbo2 = 0.0
    
    for x1, x2, y1_true, y2_true in data:
        z1 = x1 * w11 + x2 * w21 + bh1
        z2 = x1 * w12 + x2 * w22 + bh2
        z3 = x1 * w13 + x2 * w23 + bh3
        
        h1 = sigmoid(z1)
        h2 = sigmoid(z2)
        h3 = sigmoid(z3)
        
        zo1 = h1 * w31 + h2 * w41 + h3 * w51 + bo1
        zo2 = h1 * w32 + h2 * w42 + h3 * w52 + bo2
        
        y1_hat = sigmoid(zo1)
        y2_hat = sigmoid(zo2)
        
        e1 = y1_true - y1_hat
        e2 = y2_true - y2_hat
        total_error += (e1**2 + e2**2)
        
        dy1 = e1 * y1_hat * (1 - y1_hat)
        dy2 = e2 * y2_hat * (1 - y2_hat)
        
        dbo1 += dy1
        dbo2 += dy2
        
        dw31 += dy1 * h1
        dw32 += dy2 * h1
        dw41 += dy1 * h2
        dw42 += dy2 * h2
        dw51 += dy1 * h3
        dw52 += dy2 * h3
        
        dh1 = dy1 * w31 + dy2 * w32
        dh2 = dy1 * w41 + dy2 * w42
        dh3 = dy1 * w51 + dy2 * w52
        
        dz1 = dh1 * h1 * (1 - h1)
        dz2 = dh2 * h2 * (1 - h2)
        dz3 = dh3 * h3 * (1 - h3)
        
        dbh1 += dz1
        dbh2 += dz2
        dbh3 += dz3
        
        dw11 += dz1 * x1
        dw21 += dz1 * x2
        dw12 += dz2 * x1
        dw22 += dz2 * x2
        dw13 += dz3 * x1
        dw23 += dz3 * x2
    
    w11 += n * dw11
    w12 += n * dw12
    w13 += n * dw13
    w21 += n * dw21
    w22 += n * dw22
    w23 += n * dw23
    
    w31 += n * dw31
    w32 += n * dw32
    w41 += n * dw41
    w42 += n * dw42
    w51 += n * dw51
    w52 += n * dw52
    
    bh1 += n * dbh1
    bh2 += n * dbh2
    bh3 += n * dbh3
    bo1 += n * dbo1
    bo2 += n * dbo2
    
    return total_error

def train(epochs=300):
    print("Initial weights and biases:")
    print(f"w11={w11:.4f} w12={w12:.4f} w13={w13:.4f}")
    print(f"w21={w21:.4f} w22={w22:.4f} w23={w23:.4f}")
    print(f"w31={w31:.4f} w32={w32:.4f}")
    print(f"w41={w41:.4f} w42={w42:.4f}")
    print(f"w51={w51:.4f} w52={w52:.4f}")
    print(f"bh1={bh1:.4f} bh2={bh2:.4f} bh3={bh3:.4f}")
    print(f"bo1={bo1:.4f}  bo2={bo2:.4f}\n")
    
    for epoch in range(epochs):
        error_sum = update_batch()
        
        if epoch % 30 == 0 or epoch == epochs-1:
            y1p, y2p = predict(0.5, 0.8)
            print(f"Epoch {epoch+1:4d} | error = {error_sum:8.6f} | "
                  f"y1={y1p:.6f} y2={y2p:.6f}  (targets 0.700000 0.400000)")
    
    print("\nFinal weights and biases:")
    print(f"w11={w11:.4f} w12={w12:.4f} w13={w13:.4f}")
    print(f"w21={w21:.4f} w22={w22:.4f} w23={w23:.4f}")
    print(f"w31={w31:.4f} w32={w32:.4f}")
    print(f"w41={w41:.4f} w42={w42:.4f}")
    print(f"w51={w51:.4f} w52={w52:.4f}")
    print(f"bh1={bh1:.4f} bh2={bh2:.4f} bh3={bh3:.4f}")
    print(f"bo1={bo1:.4f}  bo2={bo2:.4f}")
    
    final_y1, final_y2 = predict(0.5, 0.8)
    print(f"\nFinal prediction: y1 = {final_y1:.6f}  y2 = {final_y2:.6f}")

train(300)

# B. Abdelkader – 2026 😊