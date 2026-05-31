from math import exp
import copy
import csv

# 1. Global state - all learnable parameters live here
W = {
    'w11': 0.1, 'w12': 0.2, 'w13': 0.3,
    'w21': 0.4, 'w22': 0.5, 'w23': 0.6,
    'w31': 0.1, 'w32': 0.4,
    'w41': 0.2, 'w42': 0.5,
    'w51': 0.3, 'w52': 0.6,
    'bh1': 0.7, 'bh2': 0.8, 'bh3': 0.9,
    'bo1': 0.1, 'bo2': 0.2,
}

# History will collect snapshots after each update
history = []

# 2. Pure functions

def sigmoid(z):
    return 1 / (1 + exp(-z))

def forward(x1, x2):
    z1 = x1 * W['w11'] + x2 * W['w21'] + W['bh1']
    z2 = x1 * W['w12'] + x2 * W['w22'] + W['bh2']
    z3 = x1 * W['w13'] + x2 * W['w23'] + W['bh3']
    
    h1 = sigmoid(z1)
    h2 = sigmoid(z2)
    h3 = sigmoid(z3)
    
    zo1 = h1 * W['w31'] + h2 * W['w41'] + h3 * W['w51'] + W['bo1']
    zo2 = h1 * W['w32'] + h2 * W['w42'] + h3 * W['w52'] + W['bo2']
    
    y1_hat = sigmoid(zo1)
    y2_hat = sigmoid(zo2)
    
    cache = {
        'x1': x1, 'x2': x2,
        'z1': z1, 'z2': z2, 'z3': z3,
        'h1': h1, 'h2': h2, 'h3': h3,
        'zo1': zo1, 'zo2': zo2,
        'y1_hat': y1_hat, 'y2_hat': y2_hat
    }
    return (y1_hat, y2_hat), cache


def mse_loss(y1_true, y2_true, y1_pred, y2_pred):
    e1 = y1_true - y1_pred
    e2 = y2_true - y2_pred
    return (e1**2 + e2**2) / 2

def compute_gradients(y1_true, y2_true, cache):
    y1_hat = cache['y1_hat']
    y2_hat = cache['y2_hat']
    h1, h2, h3 = cache['h1'], cache['h2'], cache['h3']
    
    dy1 = (y1_hat - y1_true) * y1_hat * (1 - y1_hat)
    dy2 = (y2_hat - y2_true) * y2_hat * (1 - y2_hat)
    
    grads = {}
    
    grads['bo1'] = dy1
    grads['bo2'] = dy2
    
    grads['w31'] = dy1 * h1
    grads['w32'] = dy2 * h1
    grads['w41'] = dy1 * h2
    grads['w42'] = dy2 * h2
    grads['w51'] = dy1 * h3
    grads['w52'] = dy2 * h3
    
    dh1 = dy1 * W['w31'] + dy2 * W['w32']
    dh2 = dy1 * W['w41'] + dy2 * W['w42']
    dh3 = dy1 * W['w51'] + dy2 * W['w52']
    
    dz1 = dh1 * h1 * (1 - h1)
    dz2 = dh2 * h2 * (1 - h2)
    dz3 = dh3 * h3 * (1 - h3)
    
    grads['bh1'] = dz1
    grads['bh2'] = dz2
    grads['bh3'] = dz3
    
    grads['w11'] = dz1 * cache['x1']
    grads['w21'] = dz1 * cache['x2']
    grads['w12'] = dz2 * cache['x1']
    grads['w22'] = dz2 * cache['x2']
    grads['w13'] = dz3 * cache['x1']
    grads['w23'] = dz3 * cache['x2']
    
    return grads


def update_weights(grads, lr=0.1, step=0, y1_pred=0.0, y2_pred=0.0, y1_true=0.0, y2_true=0.0):
    for key in W:
        W[key] -= lr * grads.get(key, 0.0)    
    # Create snapshot with extra performance fields
    snapshot = copy.deepcopy(W)
    snapshot['step']     = step
    snapshot['y1_pred']  = y1_pred
    snapshot['y2_pred']  = y2_pred
    snapshot['err1']     = y1_true - y1_pred
    snapshot['err2']     = y2_true - y2_pred
    snapshot['loss']     = mse_loss(y1_true, y2_true, y1_pred, y2_pred)
    
    history.append(snapshot)


# Main training loop

x1, x2       = 0.5, 0.8
y1_true, y2_true = 0.7, 0.4
lr = 0.1
n_steps = 20   

print("Starting training...\n")

for step in range(n_steps):
    (y1p, y2p), cache = forward(x1, x2)
    loss = mse_loss(y1_true, y2_true, y1p, y2p)
    
    print(f"Step {step+1:2d} | y1={y1p:6.4f}  y2={y2p:6.4f}  loss={loss:.6f}")
    
    grads = compute_gradients(y1_true, y2_true, cache)
    update_weights(grads, lr, step+1, y1p, y2p, y1_true, y2_true)


for row in history:
    values = [row.get(k, 0.0) for k in [
        'step','loss','y1_pred','y2_pred','err1','err2',
        'bh1','bh2','bh3','bo1','bo2',
        'w11','w12','w13','w21','w22','w23',
        'w31','w32','w41','w42','w51','w52'
    ]]
    print(",".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in values))

with open('hist_train.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader()
    writer.writerows(history)