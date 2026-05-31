
import math as m
import random as r

# hyperpaameters nu 
n = 0.1                 

# ncomment thus section if you want to tr random initialization weight and bias.
#w = [r.uniform(-1.0, 1.0), r.uniform(-1.0, 1.0)]
#b = r.uniform(-1.0, 1.0)             
w = [.1, .1]
b = -.2
data = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]
err = [] 
# to execte the model!
def predict(x1, x2):
    z = x1 * w[0] + x2 * w[1] + b
    return 1 if z > 0 else 0    
#note that the perceptron in t's core concept have the linear activation function f(z) = z,
#but we can use other activation functions to make it more powerful.
def activation(z):
#    return 1 if z > 0 else 0       # it take onl 1 iteration to converge with w = [.1, .1] and b = -.2 (o whatever initialization)
#    return 1 / (1 + m.exp(-z))     #  Sigmoid activation with initial weight w = [.1, .1] take 147 iteration to converge!
#    return max(0, z)               # ReLU activation divege even with 1000000 iteration and w = [.1, .1] and b = -.2 (o whatever initialization)

    return z                        # Diverge final result  Epoch 99999 | error =  0.250 | w = [ 0.500,  0.500]  b = -0.250
                                    #    diverge start from 
                                    #    Epoch 180 | error =  0.250 | w = [ 0.486,  0.486]  b = -0.233 
def update_batch():
    global w, b
    dw0_total = 0
    dw1_total = 0
    db_total   = 0
    total_error = 0
    for x1, x2, target in data:
        z = x1 * w[0] + x2 * w[1] + b
        y = activation(z)
        error = target - y           
        total_error += error ** 2
        
        dw0_total += error * x1
        dw1_total += error * x2
        db_total   += error
    
    dw0 = n * (dw0_total / 4)
    dw1 = n * (dw1_total / 4)
    db  = n * (db_total  / 4)
    
    w[0] += dw0
    w[1] += dw1
    b    += db
    
    return total_error


def train(epochs=20):
    print("Initial weights:", w, "bias:", b, "\n")
    
    for epoch in range(epochs):
        error_sum = update_batch()
        err.append(error_sum)
        
        if epoch % 5 == 0 or epoch == epochs-1:
            print(f"Epoch {epoch+1:3d} | error = {error_sum:6.3f} | "
                  f"w = [{w[0]:6.3f}, {w[1]:6.3f}]  b = {b:6.3f}")
            
        
        # Early stopping if error is very low    
        if error_sum < 0.001:
            print("\nConverged!")
            break          
    print("\nFinal weights:", [round(x,4) for x in w], "bias:", round(b,4))
    print("\nFinal predictions:")

train(147)
for x1, x2, target in data:
    pred = predict(x1, x2)
    print(f"{x1},{x2} → {pred}  (target = {target})")

# B. Abdelkader All rights reserved. 2026 😊